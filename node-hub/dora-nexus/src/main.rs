//! dora-nexus: OpenArm URDF renderer with live point cloud overlay.
//!
//! Renders the robot at given joint angles with optional live RealSense
//! point cloud streaming for grasp pose validation.
//!
//! Usage:
//!   # Render at home position, save screenshot
//!   dora-nexus --urdf openarm_v10.urdf --screenshot /tmp/robot.png
//!
//!   # Render with joint angles (JSON)
//!   dora-nexus --urdf openarm_v10.urdf \
//!     --joints '{"L_J1": 0.5, "L_J2": -0.3, "L_J4": 1.0}' \
//!     --screenshot /tmp/grasp_pose.png
//!
//!   # Render from a trajectory JSON file at a specific waypoint
//!   dora-nexus --urdf openarm_v10.urdf \
//!     --trajectory trajectory.json --waypoint 100 \
//!     --screenshot /tmp/waypoint_100.png
//!
//!   # Live point cloud from openarm-config.json
//!   dora-nexus --urdf openarm_v10.urdf --config openarm-config.json
//!
//!   # Trajectory + live point cloud
//!   dora-nexus --urdf openarm_v10.urdf --config openarm-config.json \
//!     --trajectory trajectory.json --waypoint 100

mod urdf;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use kiss3d::camera::OrbitCamera3d;
use kiss3d::color::Color;
use kiss3d::glamx::{EulerRot, Quat, Vec3};
use kiss3d::scene::SceneNode3d;
use kiss3d::window::Window;
use serde::Deserialize;
use xoq::realsense_client::{Intrinsics, RealSenseFrames, SyncRealSenseClient};

use base64::Engine;
use urdf::{build_urdf_scene, clean_urdf_string, compute_fk_positions, set_joint_angles};

const LEFT_JOINTS: [&str; 7] = ["L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_J7"];
const RIGHT_JOINTS: [&str; 7] = ["R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_J7"];

// --- Config structs for openarm-config.json ---

#[derive(Deserialize)]
struct OpenArmConfig {
    #[serde(rename = "armPairs")]
    arm_pairs: Vec<ArmPairConfig>,
    realsense: Vec<RealSenseConfig>,
}

#[derive(Deserialize)]
struct ArmPairConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    position: Pos3,
    rotation: Rot3,
}

#[derive(Deserialize)]
struct RealSenseConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    path: String,
    #[serde(default)]
    label: String,
    position: Pos3,
    rotation: Rot3,
    #[serde(rename = "pointSize", default = "default_point_size")]
    point_size: f32,
}

#[derive(Deserialize, Clone, Copy)]
struct Pos3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Deserialize, Clone, Copy)]
struct Rot3 {
    roll: f32,
    pitch: f32,
    yaw: f32,
}

fn default_true() -> bool {
    true
}
fn default_point_size() -> f32 {
    2.0
}

/// Camera extrinsic rotation from config degrees (roll, pitch, yaw).
/// Convention: Ry(yaw) * Rx(roll) * Ry(pitch)
fn config_rotation_quat(rot: &Rot3) -> Quat {
    let roll = rot.roll.to_radians();
    let pitch = rot.pitch.to_radians();
    let yaw = rot.yaw.to_radians();
    let ry_yaw = Quat::from_euler(EulerRot::YXZ, yaw, 0.0, 0.0);
    let rx_roll = Quat::from_euler(EulerRot::YXZ, 0.0, roll, 0.0);
    let ry_pitch = Quat::from_euler(EulerRot::YXZ, pitch, 0.0, 0.0);
    ry_yaw * rx_roll * ry_pitch
}

// --- RealSense streaming ---

struct CameraStream {
    latest_frame: Arc<Mutex<Option<RealSenseFrames>>>,
    latest_intrinsics: Arc<Mutex<Option<Intrinsics>>>,
    position: Vec3,
    rotation: Quat,
    point_size: f32,
}

fn spawn_realsense_subscriber(cfg: &RealSenseConfig) -> CameraStream {
    let latest_frame: Arc<Mutex<Option<RealSenseFrames>>> = Arc::new(Mutex::new(None));
    let latest_intrinsics: Arc<Mutex<Option<Intrinsics>>> = Arc::new(Mutex::new(None));

    let frame_slot = latest_frame.clone();
    let intr_slot = latest_intrinsics.clone();
    let path = cfg.path.clone();
    let label = cfg.label.clone();

    thread::spawn(move || {
        let display = if label.is_empty() { &path } else { &label };
        println!("[dora-nexus] RealSense: connecting to '{display}' ...");
        let mut client = match SyncRealSenseClient::connect_auto(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[dora-nexus] RealSense: failed to connect to '{display}': {e}");
                return;
            }
        };
        println!("[dora-nexus] RealSense: streaming from '{display}'");
        loop {
            match client.read_frames() {
                Ok(frames) => {
                    if let Some(intr) = client.intrinsics() {
                        *intr_slot.lock().unwrap() = Some(intr);
                    }
                    *frame_slot.lock().unwrap() = Some(frames);
                }
                Err(e) => {
                    eprintln!("[dora-nexus] RealSense: read error on '{display}': {e}");
                    break;
                }
            }
        }
    });

    CameraStream {
        latest_frame,
        latest_intrinsics,
        position: Vec3::new(cfg.position.x, cfg.position.y, cfg.position.z),
        rotation: config_rotation_quat(&cfg.rotation),
        point_size: cfg.point_size,
    }
}

// --- CLI ---

struct CliConfig {
    urdf_path: PathBuf,
    joints: HashMap<String, f32>,
    screenshot_path: Option<String>,
    camera_pos: Vec3,
    camera_target: Vec3,
    openarm_config_path: Option<String>,
    show_collisions: bool,
}

fn parse_args() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut urdf_path = PathBuf::from("examples/openarm/openarm_v10.urdf");
    let mut joints: HashMap<String, f32> = HashMap::new();
    let mut screenshot_path: Option<String> = None;
    let mut camera_pos = Vec3::new(0.8, 0.6, 0.8);
    let mut camera_target = Vec3::new(-0.1, 0.3, -0.1);
    let mut trajectory_path: Option<String> = None;
    let mut waypoint: usize = 0;
    let mut openarm_config_path: Option<String> = None;
    let mut show_collisions = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--urdf" => {
                i += 1;
                urdf_path = PathBuf::from(&args[i]);
            }
            "--joints" => {
                i += 1;
                if let Ok(map) = serde_json::from_str::<HashMap<String, f32>>(&args[i]) {
                    joints = map;
                } else {
                    eprintln!("Warning: failed to parse --joints JSON");
                }
            }
            "--trajectory" => {
                i += 1;
                trajectory_path = Some(args[i].clone());
            }
            "--waypoint" | "-w" => {
                i += 1;
                waypoint = args[i].parse().unwrap_or(0);
            }
            "--screenshot" | "-s" => {
                i += 1;
                screenshot_path = Some(args[i].clone());
            }
            "--config" | "-c" => {
                i += 1;
                openarm_config_path = Some(args[i].clone());
            }
            "--camera-pos" => {
                i += 1;
                let parts: Vec<f32> = args[i].split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();
                if parts.len() == 3 {
                    camera_pos = Vec3::new(parts[0], parts[1], parts[2]);
                }
            }
            "--camera-target" => {
                i += 1;
                let parts: Vec<f32> = args[i].split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();
                if parts.len() == 3 {
                    camera_target = Vec3::new(parts[0], parts[1], parts[2]);
                }
            }
            "--collision" => {
                show_collisions = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: dora-nexus [OPTIONS]");
                eprintln!("  --urdf <path>           URDF file path");
                eprintln!("  --joints <json>         Joint angles as JSON object");
                eprintln!("  --trajectory <path>     Trajectory JSON file");
                eprintln!("  --waypoint <n>          Waypoint index in trajectory");
                eprintln!("  --config <path>         OpenArm config JSON (enables live point cloud)");
                eprintln!("  --screenshot <path>     Save screenshot and exit");
                eprintln!("  --camera-pos <x,y,z>    Camera position");
                eprintln!("  --camera-target <x,y,z> Camera look-at target");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // Load trajectory if specified
    if let Some(traj_path) = trajectory_path {
        if let Ok(content) = std::fs::read_to_string(&traj_path) {
            if let Ok(traj_json) = serde_json::from_str::<serde_json::Value>(&content) {
                // Detect format: v3 CAN frames (has "commands") vs raw waypoints
                let arm_from_metadata = traj_json
                    .get("metadata").and_then(|m| m.get("arm")).and_then(|v| v.as_str());
                let arm_from_root = traj_json.get("arm").and_then(|v| v.as_str());
                let arm = arm_from_metadata.or(arm_from_root).unwrap_or("left");
                let joint_names = if arm == "left" { &LEFT_JOINTS } else { &RIGHT_JOINTS };

                if let Some(commands) = traj_json.get("commands").and_then(|v| v.as_array()) {
                    // v3 format: base64-encoded 72-byte CAN wire frames
                    let num_commands = commands.len();
                    let wp_idx = waypoint.min(num_commands.saturating_sub(1));
                    if let Some(cmd) = commands.get(wp_idx) {
                        if let Some(frames) = cmd.get("frames").and_then(|v| v.as_array()) {
                            let b64 = base64::engine::general_purpose::STANDARD;
                            for (j, frame) in frames.iter().take(7).enumerate() {
                                if let Some(data_b64) = frame.get("data").and_then(|v| v.as_str()) {
                                    if let Ok(wire) = b64.decode(data_b64) {
                                        if wire.len() >= 10 {
                                            // Position in payload[0..2] (wire[8..10])
                                            let raw = ((wire[8] as u16) << 8) | wire[9] as u16;
                                            let pos = raw as f32 / 65535.0 * 25.0 - 12.5;
                                            joints.insert(joint_names[j].to_string(), pos);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    println!(
                        "[dora-nexus] Loaded v3 trajectory: {} arm, waypoint {}/{} (t={:.1}s)",
                        arm, wp_idx, num_commands,
                        commands.get(wp_idx)
                            .and_then(|c| c.get("t"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0)
                    );
                } else if let Some(waypoints) = traj_json.get("waypoints").and_then(|v| v.as_array()) {
                    // Raw waypoint format: array of joint angle arrays
                    if let Some(wp) = waypoints.get(waypoint) {
                        if let Some(angles) = wp.as_array() {
                            for (j, name) in joint_names.iter().enumerate() {
                                if let Some(val) = angles.get(j).and_then(|v| v.as_f64()) {
                                    joints.insert(name.to_string(), val as f32);
                                }
                            }
                        }
                    }
                    println!("[dora-nexus] Loaded trajectory: {} arm, waypoint {}", arm, waypoint);
                }
            }
        }
    }

    CliConfig {
        urdf_path,
        joints,
        screenshot_path,
        camera_pos,
        camera_target,
        openarm_config_path,
        show_collisions,
    }
}

#[kiss3d::main]
async fn main() {
    let config = parse_args();

    if !config.urdf_path.exists() {
        eprintln!("URDF file not found: {}", config.urdf_path.display());
        std::process::exit(1);
    }

    let urdf_dir = config.urdf_path.parent().unwrap();
    let urdf_string = std::fs::read_to_string(&config.urdf_path)
        .expect("Failed to read URDF");
    let urdf_clean = clean_urdf_string(&urdf_string);
    let robot = urdf_rs::read_from_string(&urdf_clean)
        .expect("Failed to parse URDF");

    println!(
        "[dora-nexus] Loaded '{}' ({} links, {} joints)",
        robot.name, robot.links.len(), robot.joints.len()
    );

    // Load openarm config if provided.
    let openarm_config = config.openarm_config_path.as_ref().map(|path| {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| {
                eprintln!("Failed to read config '{path}': {e}");
                std::process::exit(1);
            });
        let cfg: OpenArmConfig = serde_json::from_str(&content)
            .unwrap_or_else(|e| {
                eprintln!("Failed to parse config: {e}");
                std::process::exit(1);
            });
        cfg
    });

    // Spawn RealSense subscribers.
    let mut camera_streams: Vec<CameraStream> = Vec::new();
    if let Some(ref oa_cfg) = openarm_config {
        for rs_cfg in &oa_cfg.realsense {
            if rs_cfg.enabled {
                camera_streams.push(spawn_realsense_subscriber(rs_cfg));
            }
        }
        println!("[dora-nexus] {} camera stream(s) active", camera_streams.len());
    }

    // Determine arm pair transform.
    let arm_pair_pos;
    let arm_pair_rot;
    if let Some(ref oa_cfg) = openarm_config {
        if let Some(pair) = oa_cfg.arm_pairs.iter().find(|p| p.enabled) {
            arm_pair_pos = Vec3::new(pair.position.x, pair.position.y, pair.position.z);
            arm_pair_rot = config_rotation_quat(&pair.rotation);
        } else {
            arm_pair_pos = Vec3::ZERO;
            arm_pair_rot = Quat::IDENTITY;
        }
    } else {
        arm_pair_pos = Vec3::ZERO;
        arm_pair_rot = Quat::IDENTITY;
    }

    // URDF Z-up to kiss3d Y-up.
    let urdf_to_yup = Quat::from_euler(EulerRot::XYZ, -std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    let robot_root_rot = arm_pair_rot * urdf_to_yup;
    let robot_root_pos = arm_pair_pos;

    // Set up window.
    let mut window = Window::new("dora-nexus").await;
    window.set_background_color(Color::new(0.12, 0.12, 0.18, 1.0));

    let mut scene = SceneNode3d::empty();
    scene.add_directional_light(Vec3::new(-1.0, -1.0, -1.0));
    scene.add_directional_light(Vec3::new(1.0, 1.0, 1.0));
    scene.add_directional_light(Vec3::new(0.0, 1.0, 0.0));

    // Build URDF scene with arm pair + Y-up transform.
    let mut robot_root = SceneNode3d::empty();
    robot_root.set_position(robot_root_pos);
    robot_root.set_rotation(robot_root_rot);
    let mut link_nodes = build_urdf_scene(&robot, urdf_dir, &mut robot_root);
    scene.add_child(robot_root);

    // Apply joint angles.
    if !config.joints.is_empty() {
        set_joint_angles(&robot, &mut link_nodes, &config.joints);
        println!("[dora-nexus] Applied {} joint angles", config.joints.len());
    }

    let mut camera = OrbitCamera3d::new(config.camera_pos, config.camera_target);
    let mut frame: u64 = 0;
    let has_streams = !camera_streams.is_empty();

    let step: usize = 4;
    let max_depth_m: f32 = 2.0;
    let collision_threshold: f32 = 0.05;
    let mut last_collision_log: u64 = 0;
    let mut screenshot_taken = false;
    let mut first_pc_frame: Option<u64> = None; // frame when first point cloud data was drawn
    let show_collisions = config.show_collisions;

    // Per-camera frame cache.
    let mut current_frames: Vec<Option<RealSenseFrames>> = (0..camera_streams.len()).map(|_| None).collect();
    let mut current_intrinsics: Vec<Option<Intrinsics>> = (0..camera_streams.len()).map(|_| None).collect();

    while window.render_3d(&mut scene, &mut camera).await {
        // Compute FK for collision detection.
        let joint_spheres = compute_fk_positions(&robot, &config.joints, robot_root_pos, robot_root_rot);

        // Draw collision spheres.
        if show_collisions {
            for (_, pos, radius) in &joint_spheres {
                window.draw_point(*pos, Color::new(0.0, 1.0, 1.0, 0.5), 4.0);
                let r = *radius;
                window.draw_line(*pos - Vec3::X * r, *pos + Vec3::X * r, Color::new(0.0, 1.0, 1.0, 0.3), 1.0, false);
                window.draw_line(*pos - Vec3::Y * r, *pos + Vec3::Y * r, Color::new(0.0, 1.0, 1.0, 0.3), 1.0, false);
                window.draw_line(*pos - Vec3::Z * r, *pos + Vec3::Z * r, Color::new(0.0, 1.0, 1.0, 0.3), 1.0, false);
            }
        }

        // Render point clouds.
        let mut collision_count: u32 = 0;
        let mut closest_link: Option<(String, f32)> = None;
        let mut have_pointcloud = false;

        for (i, stream) in camera_streams.iter().enumerate() {
            {
                let mut guard = stream.latest_frame.lock().unwrap();
                if guard.is_some() {
                    current_frames[i] = guard.take();
                }
            }
            {
                let guard = stream.latest_intrinsics.lock().unwrap();
                if guard.is_some() {
                    current_intrinsics[i] = *guard;
                }
            }

            let (Some(frames), Some(intr)) = (&current_frames[i], &current_intrinsics[i]) else {
                continue;
            };
            have_pointcloud = true;

            let w = frames.width as usize;
            let h = frames.height as usize;

            // Debug: log frame stats on first frame.
            if first_pc_frame.is_none() {
                println!(
                    "[dora-nexus] Frame: {}x{}, depth_mm len={}, color_rgb len={}, intr: fx={:.1} fy={:.1} ppx={:.1} ppy={:.1}",
                    w, h, frames.depth_mm.len(), frames.color_rgb.len(),
                    intr.fx, intr.fy, intr.ppx, intr.ppy
                );
            }

            let mut point_count: u32 = 0;
            let mut min_pt = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            let mut max_pt = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

            for v in (0..h).step_by(step) {
                for u in (0..w).step_by(step) {
                    let idx = v * w + u;
                    let depth_mm = frames.depth_mm[idx];
                    if depth_mm == 0 {
                        continue;
                    }

                    let d = depth_mm as f32 / 1000.0;
                    if d > max_depth_m {
                        continue;
                    }

                    // Unproject: negate X and Y to convert image coords to point cloud frame.
                    let cx = -((u as f32 - intr.ppx) * d / intr.fx);
                    let cy = -((v as f32 - intr.ppy) * d / intr.fy);
                    let cz = d;

                    // Apply camera extrinsic transform.
                    let cam_pt = Vec3::new(cx, cy, cz);
                    let rotated = stream.rotation * cam_pt;
                    let world_pt = rotated + stream.position;

                    // Check collision against robot joint spheres.
                    let mut colliding = false;
                    if show_collisions {
                        for (link_name, joint_pos, radius) in &joint_spheres {
                            let dist = (world_pt - *joint_pos).length();
                            if dist < *radius + collision_threshold {
                                colliding = true;
                                collision_count += 1;
                                let effective_dist = dist - radius;
                                if closest_link.as_ref().map_or(true, |(_, d)| effective_dist < *d) {
                                    closest_link = Some((link_name.clone(), effective_dist));
                                }
                                break;
                            }
                        }
                    }

                    point_count += 1;
                    min_pt = min_pt.min(world_pt);
                    max_pt = max_pt.max(world_pt);

                    let ci = idx * 3;
                    if ci + 2 >= frames.color_rgb.len() {
                        continue;
                    }
                    if colliding {
                        window.draw_point(world_pt, Color::new(1.0, 0.0, 0.0, 1.0), stream.point_size * 2.0);
                    } else {
                        let r = frames.color_rgb[ci] as f32 / 255.0;
                        let g = frames.color_rgb[ci + 1] as f32 / 255.0;
                        let b = frames.color_rgb[ci + 2] as f32 / 255.0;
                        window.draw_point(world_pt, Color::new(r, g, b, 1.0), stream.point_size);
                    }
                }
            }

            if first_pc_frame.is_none() && point_count > 0 {
                println!(
                    "[dora-nexus] Points: {} rendered, bounds: ({:.2},{:.2},{:.2}) → ({:.2},{:.2},{:.2})",
                    point_count, min_pt.x, min_pt.y, min_pt.z, max_pt.x, max_pt.y, max_pt.z
                );
            }
        }

        // Log collisions (throttled).
        if collision_count > 0 && frame - last_collision_log >= 60 {
            if let Some((link_name, dist)) = &closest_link {
                println!(
                    "[dora-nexus] COLLISION: {} points colliding! Closest: '{}' ({:.3}m)",
                    collision_count, link_name, dist
                );
            }
            last_collision_log = frame;
        }

        // Draw world axes.
        let o = Vec3::ZERO;
        window.draw_line(o, Vec3::new(0.3, 0.0, 0.0), Color::new(1.0, 0.0, 0.0, 1.0), 2.0, false);
        window.draw_line(o, Vec3::new(0.0, 0.3, 0.0), Color::new(0.0, 1.0, 0.0, 1.0), 2.0, false);
        window.draw_line(o, Vec3::new(0.0, 0.0, 0.3), Color::new(0.0, 0.0, 1.0, 1.0), 2.0, false);

        // Track when first point cloud data arrives.
        if have_pointcloud && first_pc_frame.is_none() {
            first_pc_frame = Some(frame);
        }

        // Screenshot logic:
        // draw_point calls are rendered on the NEXT render_3d call, so we must
        // wait at least one frame after the first point cloud draw before snapping.
        // - With point cloud: snap 2 frames after first data arrives.
        // - Without point cloud: snap at frame 5 (let GPU settle).
        if !screenshot_taken {
            if let Some(ref path) = config.screenshot_path {
                let ready = if has_streams {
                    first_pc_frame.map_or(false, |f| frame >= f + 2)
                } else {
                    frame >= 5
                };
                if ready {
                    let image = window.snap_image();
                    image.save(path).expect("Failed to save screenshot");
                    println!("[dora-nexus] Screenshot saved: {}", path);
                    println!(
                        "[dora-nexus] Collision summary: {} points colliding",
                        collision_count
                    );
                    if let Some((link_name, dist)) = &closest_link {
                        println!(
                            "[dora-nexus] Closest collision: '{}' ({:.3}m)",
                            link_name, dist
                        );
                    }
                    screenshot_taken = true;
                    if !has_streams || std::env::args().any(|a| a == "--auto-exit") {
                        break;
                    }
                }
            }
        }

        frame += 1;
    }
}
