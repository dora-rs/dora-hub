use std::{env, time::Duration};

use dora_node_api::{
    arrow::array::{Float32Array, UInt8Array},
    DoraNode, Event,
};
use eyre::Context;
use xoq::socketcan::{self, CanFrame, RemoteCanSocket};

const CANFD_FRAME_SIZE: usize = 72;
const NUM_MOTORS: usize = 8;

const ENABLE_MIT: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC];
const DISABLE_MIT: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD];

/// Zero-torque MIT command (holds position with zero kp/kd/torque).
/// Equivalent to `encode_mit_command(0, 0, 0, 0, 0)` from dora-openarm.
/// Position = 0x7FFF (midpoint), all other fields = 0.
const QUERY_CMD: [u8; 8] = [0x7F, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Decode a 72-byte Linux `struct canfd_frame` wire format into (can_id, payload).
fn decode_wire_frame(wire: &[u8]) -> (u32, &[u8]) {
    let can_id = u32::from_le_bytes([wire[0], wire[1], wire[2], wire[3]]) & 0x1FFFFFFF;
    let len = (wire[4] as usize).min(64);
    (can_id, &wire[8..8 + len])
}

/// Decode motor response position from 8-byte MIT protocol response.
/// Bytes [1:3] encode position as u16 in range [-12.5, 12.5] rad.
fn decode_response_pos(data: &[u8]) -> f32 {
    let raw = ((data[1] as u16) << 8) | data[2] as u16;
    raw as f32 / 65535.0 * 25.0 - 12.5
}

/// Convert anyhow::Result to eyre::Result.
fn anyhow_to_eyre<T>(r: anyhow::Result<T>) -> eyre::Result<T> {
    r.map_err(|e| eyre::eyre!("{:#}", e))
}

/// Drain all pending response frames, updating motor positions.
fn drain_responses(socket: &mut RemoteCanSocket, positions: &mut [f32; NUM_MOTORS]) {
    while let Ok(Some(frame)) = socket.read_frame() {
        let rid = frame.id();
        if (0x11..=0x18).contains(&rid) {
            let data = match &frame {
                xoq::socketcan::AnyCanFrame::Can(f) => f.data(),
                xoq::socketcan::AnyCanFrame::CanFd(f) => f.data(),
            };
            if data.len() >= 3 {
                let idx = (rid - 0x11) as usize;
                positions[idx] = decode_response_pos(data);
            }
        }
    }
}

/// Run enable sequence: ENABLE_MIT + zero-torque QUERY_CMD for each motor.
fn enable_motors(socket: &mut RemoteCanSocket) -> eyre::Result<()> {
    for motor_id in 0x01..=0x08u32 {
        let frame = anyhow_to_eyre(CanFrame::new(motor_id, &ENABLE_MIT))?;
        anyhow_to_eyre(socket.write_frame(&frame))?;
        let _ = socket.read_frame();
        let frame = anyhow_to_eyre(CanFrame::new(motor_id, &QUERY_CMD))?;
        anyhow_to_eyre(socket.write_frame(&frame))?;
        let _ = socket.read_frame();
    }
    tracing::info!("Motors enabled (0x01-0x08)");
    Ok(())
}

/// Run disable sequence: DISABLE_MIT for each motor.
fn disable_motors(socket: &mut RemoteCanSocket) -> eyre::Result<()> {
    for motor_id in 0x01..=0x08u32 {
        let frame = anyhow_to_eyre(CanFrame::new(motor_id, &DISABLE_MIT))?;
        anyhow_to_eyre(socket.write_frame(&frame))?;
        let _ = socket.read_frame();
    }
    tracing::info!("Motors disabled (0x01-0x08)");
    Ok(())
}

pub fn lib_main() -> eyre::Result<()> {
    let server_id = env::var("SERVER_ID")
        .wrap_err("SERVER_ID env var required (64-char hex iroh endpoint ID)")?;
    let timeout_ms: u64 = env::var("TIMEOUT_MS")
        .unwrap_or_else(|_| "100".into())
        .parse()
        .wrap_err("TIMEOUT_MS must be a number")?;
    let connect_timeout_s: u64 = env::var("CONNECT_TIMEOUT_S")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .wrap_err("CONNECT_TIMEOUT_S must be a number")?;

    tracing::info!(server_id, timeout_ms, connect_timeout_s, "Connecting to xoq-can server");

    let mut socket = anyhow_to_eyre(
        socketcan::new(&server_id)
            .timeout(Duration::from_secs(connect_timeout_s))
            .open(),
    )
    .wrap_err("Failed to connect to xoq-can server")?;

    anyhow_to_eyre(socket.set_timeout(Duration::from_millis(timeout_ms)))?;
    tracing::info!("Connected to xoq-can server");

    let (mut node, mut events) = DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata } => match id.as_str() {
                "enable" => {
                    enable_motors(&mut socket)?;
                }
                "disable" => {
                    disable_motors(&mut socket)?;
                }
                "can_frames" => {
                    let bytes = data
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .ok_or_else(|| eyre::eyre!("can_frames: expected UInt8 array"))?;
                    let raw = bytes.values().as_ref();

                    if raw.len() % CANFD_FRAME_SIZE != 0 {
                        tracing::warn!(
                            len = raw.len(),
                            "can_frames length not a multiple of {CANFD_FRAME_SIZE}, ignoring"
                        );
                        continue;
                    }

                    let mut positions = [f32::NAN; NUM_MOTORS];

                    for chunk in raw.chunks_exact(CANFD_FRAME_SIZE) {
                        let (can_id, payload) = decode_wire_frame(chunk);
                        let frame = anyhow_to_eyre(CanFrame::new(can_id, payload))
                            .wrap_err_with(|| format!("Invalid CAN frame: id=0x{can_id:X}"))?;
                        anyhow_to_eyre(socket.write_frame(&frame))?;
                    }

                    // Drain all responses after writing all frames
                    drain_responses(&mut socket, &mut positions);

                    let output = Float32Array::from(positions.to_vec());
                    node.send_output(
                        "joint_state".into(),
                        metadata.parameters,
                        output,
                    )?;
                }
                other => {
                    tracing::debug!(input = other, "Unknown input, ignoring");
                }
            },
            Event::Stop(_) => {
                tracing::info!("Stop event received, disabling motors");
                let _ = disable_motors(&mut socket);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
