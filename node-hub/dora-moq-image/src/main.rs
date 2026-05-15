//! dora-moq-image: Publishes dora RGB images to MoQ as AV1 video.
//!
//! Receives RGB image data from a dora input, encodes it to AV1 via
//! VideoToolbox (macOS) or NVENC (Linux), muxes as CMAF/fMP4, and
//! publishes on a MoQ "video" track that the openarm.html viewer
//! can subscribe to and display.
//!
//! Environment variables:
//!   MOQ_RELAY    - MoQ relay URL (default: https://cdn.1ms.ai)
//!   MOQ_PATH     - MoQ broadcast path (default: anon/debug-image)
//!   IMAGE_WIDTH  - Expected image width (default: 1280)
//!   IMAGE_HEIGHT - Expected image height (default: 720)

use dora_node_api::{self as dora, Event};
use std::time::{SystemTime, UNIX_EPOCH};
use xoq::cmaf::{parse_av1_frame, Av1CmafMuxer, CmafConfig};
use xoq::MoqBuilder;

#[cfg(target_os = "macos")]
use xoq::vtenc::VtEncoder;

#[cfg(not(target_os = "macos"))]
use xoq::nvenc_av1::NvencAv1Encoder;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn stamp(data: Vec<u8>, ms: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + data.len());
    out.extend_from_slice(&ms.to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn main() -> eyre::Result<()> {
    let relay = std::env::var("MOQ_RELAY").unwrap_or_else(|_| "https://cdn.1ms.ai".into());
    let path = std::env::var("MOQ_PATH").unwrap_or_else(|_| "anon/debug-image".into());
    let width: u32 = std::env::var("IMAGE_WIDTH")
        .unwrap_or_else(|_| "1280".into())
        .parse()?;
    let height: u32 = std::env::var("IMAGE_HEIGHT")
        .unwrap_or_else(|_| "720".into())
        .parse()?;
    let fps: u32 = 5;

    println!("[dora-moq-image] Publishing to {relay} path={path} {width}x{height}");

    // Create encoder
    #[cfg(target_os = "macos")]
    let mut encoder = VtEncoder::new(width, height, fps, 500_000)
        .map_err(|e| eyre::eyre!("VtEncoder: {e}"))?;

    #[cfg(not(target_os = "macos"))]
    let mut encoder = NvencAv1Encoder::new(width, height, fps, 500_000, false)
        .map_err(|e| eyre::eyre!("NvencAv1Encoder: {e}"))?;

    let mut muxer = Av1CmafMuxer::new(CmafConfig {
        timescale: 90000,
        ..Default::default()
    });

    // Connect to MoQ relay
    let rt = tokio::runtime::Runtime::new()?;
    let mut publisher = rt.block_on(async {
        MoqBuilder::new()
            .relay(&relay)
            .path(&path)
            .connect_publisher()
            .await
    }).map_err(|e| eyre::eyre!("MoQ connect: {e}"))?;

    let mut video_track = publisher.create_track("video");
    println!("[dora-moq-image] Connected, publishing on '{path}/video'");

    let mut init_segment: Option<Vec<u8>> = None;
    let mut frame_count: u64 = 0;

    // Start dora node
    let (_node, mut events) = dora::DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        if let Event::Input { id, data, .. } = event {
            if id.as_str() != "image" {
                continue;
            }

            let rgb_bytes: Vec<u8> = data.0.as_any()
                .downcast_ref::<dora_node_api::arrow::array::UInt8Array>()
                .map(|a| a.values().to_vec())
                .unwrap_or_default();
            let expected = (width * height * 3) as usize;
            if rgb_bytes.len() != expected {
                eprintln!(
                    "[dora-moq-image] Wrong size: {} (expected {})",
                    rgb_bytes.len(),
                    expected
                );
                continue;
            }

            // Debug: check if data is non-zero
            let nonzero = rgb_bytes.iter().filter(|&&b| b > 0).count();
            println!("[dora-moq-image] Received {expected} bytes, {nonzero} non-zero");

            // Encode the same frame multiple times to ensure encoder warmup
            // and a visible keyframe (NVENC first frame can be black)
            let n_repeats = if frame_count == 0 { 5 } else { 3 };
            for rep in 0..n_repeats {
                let timestamp_us = now_ms() * 1000;
                let wall_ms = now_ms();

                let av1_data = match encoder.encode_rgb(&rgb_bytes, timestamp_us) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("[dora-moq-image] Encode error: {e}");
                        break;
                    }
                };

                let parsed = parse_av1_frame(&av1_data);

                // Init segment on first keyframe
                if init_segment.is_none() {
                    if let Some(ref seq_hdr) = parsed.sequence_header {
                        let init = muxer.create_init_segment(seq_hdr, width, height);
                        video_track.write(init.clone());
                        init_segment = Some(init);
                        println!("[dora-moq-image] Sent AV1 CMAF init segment");
                    }
                }

                let sub_frame = frame_count * n_repeats as u64 + rep as u64;
                let pts = (sub_frame as i64) * 90000 / fps as i64;
                let dts = pts;
                let duration = (90000 / fps) as u32;

                muxer.add_frame(&parsed.data, pts, dts, duration, parsed.is_keyframe);
                if let Some(segment) = muxer.flush() {
                    if let Some(ref init) = init_segment {
                        let mut combined = init.clone();
                        combined.extend_from_slice(&segment);
                        video_track.write(combined);
                    } else {
                        video_track.write(segment);
                    }
                }
            }
            println!("[dora-moq-image] Published frame {frame_count} ({n_repeats} repeats)");

            frame_count += 1;
        }
    }

    Ok(())
}
