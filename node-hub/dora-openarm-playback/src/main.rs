//! dora-openarm-playback — Rust dora node for OpenArm CAN trajectory playback.
//!
//! Receives trajectory JSON (v2/v3 format) over dora, connects to the arm
//! via xoq::socketcan (iroh P2P), and plays back CAN frames with timing.
//!
//! Inputs:
//!   - `tick`        — query all 8 motors, publish `joint_state`
//!   - `trajectory`  — JSON string (v3 trajectory) → play back
//!   - `enable`      — enable all motors
//!   - `disable`     — disable all motors
//!
//! Outputs:
//!   - `joint_state`        — float32[8] motor positions (radians)
//!   - `trajectory_status`  — string: "playing" / "done" / "error: ..."
//!
//! Env vars:
//!   - `CAN_INTERFACE`    — 64-char hex iroh node ID (required)
//!   - `ARM_SIDE`         — "left" or "right" (default "left")
//!   - `TIMEOUT_MS`       — CAN read timeout in ms (default 100)
//!   - `CONNECT_TIMEOUT`  — iroh connection timeout seconds (default 10)
//!   - `AUTO_ENABLE`      — "true" to enable motors on startup (default "false")

use std::time::{Duration, Instant};
use std::{env, thread};

use dora_node_api::arrow::array::{Array, Float32Array, StringArray};
use dora_node_api::{DoraNode, Event};
use eyre::Context;
use xoq::socketcan::{self, CanFrame, RemoteCanSocket};

const NUM_MOTORS: usize = 8;
const CANFD_FRAME_SIZE: usize = 72;

const ENABLE_MIT: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC];
const DISABLE_MIT: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD];

// Zero-torque query: pos=0, vel=0, kp=0, kd=0, tau=0
const QUERY_CMD: [u8; 8] = [0x80, 0x00, 0x80, 0x00, 0x00, 0x00, 0x08, 0x00];

// MIT protocol range
const POS_MIN: f64 = -12.5;
const POS_MAX: f64 = 12.5;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn anyhow_to_eyre<T>(r: anyhow::Result<T>) -> eyre::Result<T> {
    r.map_err(|e| eyre::eyre!("{:#}", e))
}

fn decode_response_pos(data: &[u8]) -> f32 {
    let raw = ((data[1] as u16) << 8) | data[2] as u16;
    raw as f32 / 65535.0 * (POS_MAX - POS_MIN) as f32 + POS_MIN as f32
}

fn decode_command_pos(data: &[u8]) -> f32 {
    let raw = ((data[0] as u16) << 8) | data[1] as u16;
    raw as f32 / 65535.0 * (POS_MAX - POS_MIN) as f32 + POS_MIN as f32
}

fn decode_wire_frame(wire: &[u8]) -> (u32, &[u8]) {
    let can_id = u32::from_le_bytes([wire[0], wire[1], wire[2], wire[3]]) & 0x1FFFFFFF;
    let len = (wire[4] as usize).min(64);
    (can_id, &wire[8..8 + len])
}

fn drain_responses(socket: &mut RemoteCanSocket, positions: &mut [f32; NUM_MOTORS]) -> usize {
    let mut count = 0;
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
                count += 1;
            }
        }
    }
    count
}

fn enable_motors(socket: &mut RemoteCanSocket) -> eyre::Result<()> {
    for motor_id in 0x01..=0x08u32 {
        let frame = anyhow_to_eyre(CanFrame::new(motor_id, &ENABLE_MIT))?;
        anyhow_to_eyre(socket.write_frame(&frame))?;
        let _ = socket.read_frame();
        // Zero-torque query prevents position jump
        let frame = anyhow_to_eyre(CanFrame::new(motor_id, &QUERY_CMD))?;
        anyhow_to_eyre(socket.write_frame(&frame))?;
        let _ = socket.read_frame();
    }
    println!("Motors enabled (0x01-0x08)");
    Ok(())
}

fn disable_motors(socket: &mut RemoteCanSocket) -> eyre::Result<()> {
    for motor_id in 0x01..=0x08u32 {
        let frame = anyhow_to_eyre(CanFrame::new(motor_id, &DISABLE_MIT))?;
        anyhow_to_eyre(socket.write_frame(&frame))?;
        let _ = socket.read_frame();
    }
    println!("Motors disabled (0x01-0x08)");
    Ok(())
}

fn query_all_motors(socket: &mut RemoteCanSocket, positions: &mut [f32; NUM_MOTORS]) -> usize {
    for motor_id in 0x01..=0x08u32 {
        if let Ok(frame) = CanFrame::new(motor_id, &QUERY_CMD) {
            let _ = socket.write_frame(&frame);
        }
    }
    drain_responses(socket, positions)
}

// ---------------------------------------------------------------------------
// Trajectory JSON parsing
// ---------------------------------------------------------------------------

/// Minimal base64 decoder (no external dep, matches openarm_playback.rs).
fn base64_decode(input: &str) -> eyre::Result<Vec<u8>> {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &b in input.as_bytes() {
        if b == b'=' || b == b'\n' || b == b'\r' || b == b' ' {
            continue;
        }
        let val = TABLE
            .iter()
            .position(|&c| c == b)
            .ok_or_else(|| eyre::eyre!("invalid base64 char: {}", b as char))?
            as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

struct CanCmd {
    can_id: u32,
    data: Vec<u8>,
}

struct Timestep {
    t: f64,
    frames: Vec<CanCmd>,
}

struct Trajectory {
    arm: String,
    timesteps: Vec<Timestep>,
}

/// Parse trajectory JSON (v2/v3 format) using serde_json.
fn parse_trajectory_json(json_str: &str) -> eyre::Result<Trajectory> {
    let doc: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| eyre::eyre!("JSON parse error: {}", e))?;

    let arm = doc["metadata"]["arm"]
        .as_str()
        .unwrap_or("left")
        .to_string();

    let commands = doc["commands"]
        .as_array()
        .ok_or_else(|| eyre::eyre!("missing 'commands' array"))?;

    let mut timesteps = Vec::with_capacity(commands.len());

    for cmd in commands {
        let t = cmd["t"].as_f64().unwrap_or(0.0);
        let frames_arr = cmd["frames"]
            .as_array()
            .ok_or_else(|| eyre::eyre!("missing 'frames' in command"))?;

        let mut frames = Vec::with_capacity(frames_arr.len());
        for frame in frames_arr {
            let data_b64 = frame["data"]
                .as_str()
                .ok_or_else(|| eyre::eyre!("missing 'data' in frame"))?;
            let raw = base64_decode(data_b64)?;

            if raw.len() == CANFD_FRAME_SIZE {
                // v3: 72-byte wire frame — extract CAN ID and payload
                let (can_id, payload) = decode_wire_frame(&raw);
                frames.push(CanCmd {
                    can_id,
                    data: payload.to_vec(),
                });
            } else {
                // v2: 8-byte MIT payload, CAN ID from JSON field
                let id_str = frame["id"]
                    .as_str()
                    .ok_or_else(|| eyre::eyre!("missing 'id' in frame"))?;
                let can_id = u32::from_str_radix(id_str.trim_start_matches("0x"), 16)?;
                frames.push(CanCmd {
                    can_id,
                    data: raw,
                });
            }
        }

        timesteps.push(Timestep { t, frames });
    }

    Ok(Trajectory { arm, timesteps })
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

struct PlaybackRecord {
    t: f64,
    commanded: [f32; 7],
    actual: [f32; 7],
}

fn play_trajectory(
    socket: &mut RemoteCanSocket,
    trajectory: &Trajectory,
    positions: &mut [f32; NUM_MOTORS],
) -> eyre::Result<Vec<PlaybackRecord>> {
    if trajectory.timesteps.is_empty() {
        println!("Empty trajectory, nothing to play");
        return Ok(Vec::new());
    }

    let total = trajectory.timesteps.len();
    let duration = trajectory.timesteps.last().unwrap().t - trajectory.timesteps[0].t;
    println!(
        "Playing {} frames over {:.1}s for {} arm...",
        total, duration, trajectory.arm
    );

    let start = Instant::now();
    let t_offset = trajectory.timesteps[0].t;

    let mut commanded = [0.0f32; NUM_MOTORS];
    let mut prev_commanded = [0.0f32; NUM_MOTORS];
    let mut records = Vec::with_capacity(total);

    for (i, timestep) in trajectory.timesteps.iter().enumerate() {
        // Wait until this timestep's scheduled time
        let target_time = Duration::from_secs_f64(timestep.t - t_offset);
        let elapsed = start.elapsed();
        if target_time > elapsed {
            thread::sleep(target_time - elapsed);
        }

        // Send all CAN frames for this timestep
        for cmd in &timestep.frames {
            if let Ok(frame) = CanFrame::new(cmd.can_id, &cmd.data) {
                let _ = socket.write_frame(&frame);
            }
            // Track commanded positions
            if (0x01..=0x08).contains(&cmd.can_id) && cmd.data.len() >= 2 {
                let idx = (cmd.can_id - 0x01) as usize;
                commanded[idx] = decode_command_pos(&cmd.data);
            }
        }

        // Drain motor responses (updates positions array)
        drain_responses(socket, positions);

        // Record: previous command paired with current actual position.
        // The response to cmd[i] gives the motor's position BEFORE executing
        // cmd[i], i.e. where it settled after cmd[i-1]. This is the real
        // tracking error for the previous command (motor had one timestep to reach it).
        if i > 0 {
            let mut cmd7 = [0.0f32; 7];
            let mut act7 = [0.0f32; 7];
            cmd7.copy_from_slice(&prev_commanded[..7]);
            act7.copy_from_slice(&positions[..7]);
            records.push(PlaybackRecord {
                t: timestep.t,
                commanded: cmd7,
                actual: act7,
            });
        }
        prev_commanded.copy_from_slice(&commanded);

        // Progress log every 50 frames
        if i % 50 == 0 || i == total - 1 {
            let pct = (i + 1) as f32 / total as f32 * 100.0;
            println!(
                "  [{}/{}] {:.0}% ({:.1}s)",
                i + 1,
                total,
                pct,
                start.elapsed().as_secs_f64()
            );
        }
    }

    println!(
        "Playback complete ({:.1}s actual)",
        start.elapsed().as_secs_f64()
    );
    Ok(records)
}

// ---------------------------------------------------------------------------
// Extract string from Arrow data
// ---------------------------------------------------------------------------

fn extract_string(data: &dora_node_api::ArrowData) -> eyre::Result<String> {
    // Try StringArray (Utf8) — most common from Python pa.array(["..."])
    if let Some(arr) = data.as_any().downcast_ref::<StringArray>() {
        if arr.len() > 0 {
            return Ok(arr.value(0).to_string());
        }
    }
    // Try LargeStringArray (LargeUtf8)
    if let Some(arr) = data
        .as_any()
        .downcast_ref::<dora_node_api::arrow::array::LargeStringArray>()
    {
        if arr.len() > 0 {
            return Ok(arr.value(0).to_string());
        }
    }
    // Try raw bytes (UInt8Array)
    if let Some(arr) = data
        .as_any()
        .downcast_ref::<dora_node_api::arrow::array::UInt8Array>()
    {
        let bytes = arr.values().as_ref();
        return String::from_utf8(bytes.to_vec()).map_err(|e| eyre::eyre!("invalid UTF-8: {}", e));
    }
    Err(eyre::eyre!(
        "expected string data, got {:?}",
        data.data_type()
    ))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> eyre::Result<()> {
    let server_id = env::var("CAN_INTERFACE")
        .wrap_err("CAN_INTERFACE env var required (64-char hex iroh node ID)")?;
    let arm_side = env::var("ARM_SIDE").unwrap_or_else(|_| "left".into());
    let timeout_ms: u64 = env::var("TIMEOUT_MS")
        .unwrap_or_else(|_| "5".into())
        .parse()
        .wrap_err("TIMEOUT_MS must be a number")?;
    let connect_timeout_s: u64 = env::var("CONNECT_TIMEOUT")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .wrap_err("CONNECT_TIMEOUT must be a number")?;
    let auto_enable = matches!(
        env::var("AUTO_ENABLE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "true" | "1" | "yes"
    );

    println!(
        "dora-openarm-playback: server={}, arm={}, timeout={}ms, auto_enable={}",
        &server_id[..8.min(server_id.len())],
        arm_side,
        timeout_ms,
        auto_enable
    );

    // Connect to CAN server via iroh
    let mut socket: RemoteCanSocket = anyhow_to_eyre(
        socketcan::new(&server_id)
            .timeout(Duration::from_secs(connect_timeout_s))
            .open(),
    )
    .wrap_err("Failed to connect to CAN server")?;
    anyhow_to_eyre(socket.set_timeout(Duration::from_millis(timeout_ms)))?;
    println!("Connected to CAN server");

    if auto_enable {
        enable_motors(&mut socket)?;
    }

    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut positions = [0.0f32; NUM_MOTORS];
    let mut motors_enabled = auto_enable;
    let mut tick_count: u64 = 0;

    while let Some(event) = events.recv() {
        match event {
            Event::Input {
                id,
                data,
                metadata,
            } => match id.as_str() {
                "tick" => {
                    let n = query_all_motors(&mut socket, &mut positions);
                    tick_count += 1;
                    if tick_count % 10 == 0 {
                        println!(
                            "[tick {}] resp={}/{} pos=[{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
                            tick_count, n, NUM_MOTORS,
                            positions[0], positions[1], positions[2], positions[3],
                            positions[4], positions[5], positions[6], positions[7],
                        );
                    }
                    let output = Float32Array::from(positions.to_vec());
                    node.send_output("joint_state".into(), metadata.parameters, output)?;
                }

                "trajectory" => {
                    let json_str = extract_string(&data)?;
                    let trajectory = parse_trajectory_json(&json_str)?;

                    // Filter by arm side
                    if trajectory.arm != arm_side {
                        println!(
                            "Ignoring trajectory for {} arm (I handle {})",
                            trajectory.arm, arm_side
                        );
                        continue;
                    }

                    // Enable motors if not already enabled
                    if !motors_enabled {
                        enable_motors(&mut socket)?;
                        motors_enabled = true;
                    }

                    // Send "playing" status
                    let status = StringArray::from(vec!["playing"]);
                    node.send_output("trajectory_status".into(), Default::default(), status)?;

                    // Play back the trajectory (blocking)
                    match play_trajectory(&mut socket, &trajectory, &mut positions) {
                        Ok(records) => {
                            // Save playback log as JSON
                            if !records.is_empty() {
                                let log_data: Vec<serde_json::Value> = records.iter().map(|r| {
                                    serde_json::json!({
                                        "t": r.t,
                                        "cmd": r.commanded.to_vec(),
                                        "act": r.actual.to_vec(),
                                    })
                                }).collect();
                                let log_json = serde_json::json!({
                                    "arm": arm_side,
                                    "records": log_data,
                                });
                                let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                                let log_path = format!("trajectories/{}_{}_playback.json", ts, arm_side);
                                if let Ok(f) = std::fs::File::create(&log_path) {
                                    let _ = serde_json::to_writer(f, &log_json);
                                    println!("Playback log saved: {}", log_path);
                                }
                            }
                            let status = StringArray::from(vec!["done"]);
                            node.send_output(
                                "trajectory_status".into(),
                                Default::default(),
                                status,
                            )?;
                        }
                        Err(e) => {
                            let msg = format!("error: {}", e);
                            println!("Playback error: {}", e);
                            let status = StringArray::from(vec![msg.as_str()]);
                            node.send_output(
                                "trajectory_status".into(),
                                Default::default(),
                                status,
                            )?;
                        }
                    }
                }

                "enable" => {
                    enable_motors(&mut socket)?;
                    motors_enabled = true;
                }

                "disable" => {
                    disable_motors(&mut socket)?;
                    motors_enabled = false;
                }

                other => {
                    println!("Unknown input: {}", other);
                }
            },

            Event::Stop(_) => {
                println!("Stop event, disabling motors");
                let _ = disable_motors(&mut socket);
                break;
            }

            _ => {}
        }
    }

    Ok(())
}
