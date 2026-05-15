//! dora-nexus: OpenArm URDF renderer with screenshot capture.
//!
//! Renders the robot at given joint angles and saves a screenshot.
//! Designed for AI-in-the-loop analysis — render, screenshot, analyze.
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
//!   # Interactive mode (no --screenshot)
//!   dora-nexus --urdf openarm_v10.urdf --joints '{"L_J1": 0.5}'

mod urdf;

use std::collections::HashMap;
use std::path::PathBuf;

use kiss3d::camera::OrbitCamera3d;
use kiss3d::color::Color;
use kiss3d::glamx::Vec3;
use kiss3d::scene::SceneNode3d;
use kiss3d::window::Window;

use urdf::{build_urdf_scene, clean_urdf_string, set_joint_angles};

const LEFT_JOINTS: [&str; 7] = ["L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_J7"];
const RIGHT_JOINTS: [&str; 7] = ["R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_J7"];

struct Config {
    urdf_path: PathBuf,
    joints: HashMap<String, f32>,
    screenshot_path: Option<String>,
    camera_pos: Vec3,
    camera_target: Vec3,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut urdf_path = PathBuf::from("examples/openarm/openarm_v10.urdf");
    let mut joints: HashMap<String, f32> = HashMap::new();
    let mut screenshot_path: Option<String> = None;
    let mut camera_pos = Vec3::new(0.8, 0.6, 0.8);
    let mut camera_target = Vec3::new(-0.1, 0.3, -0.1);
    let mut trajectory_path: Option<String> = None;
    let mut waypoint: usize = 0;

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
            "--help" | "-h" => {
                eprintln!("Usage: dora-nexus [OPTIONS]");
                eprintln!("  --urdf <path>           URDF file path");
                eprintln!("  --joints <json>         Joint angles as JSON object");
                eprintln!("  --trajectory <path>     Trajectory JSON file");
                eprintln!("  --waypoint <n>          Waypoint index in trajectory");
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
                let arm = traj_json.get("arm")
                    .and_then(|v| v.as_str())
                    .unwrap_or("left");
                let num_joints = traj_json.get("num_joints")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(7) as usize;

                // Trajectory is stored as base64 CAN frames (v3) or raw waypoints
                // For simplicity, try to decode the waypoints array if present
                if let Some(waypoints) = traj_json.get("waypoints").and_then(|v| v.as_array()) {
                    if let Some(wp) = waypoints.get(waypoint) {
                        let joint_names = if arm == "left" { &LEFT_JOINTS } else { &RIGHT_JOINTS };
                        if let Some(angles) = wp.as_array() {
                            for (j, name) in joint_names.iter().enumerate() {
                                if let Some(val) = angles.get(j).and_then(|v| v.as_f64()) {
                                    joints.insert(name.to_string(), val as f32);
                                }
                            }
                        }
                    }
                }
                println!("[dora-nexus] Loaded trajectory: {} arm, waypoint {}", arm, waypoint);
            }
        }
    }

    Config {
        urdf_path,
        joints,
        screenshot_path,
        camera_pos,
        camera_target,
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

    let mut window = Window::new("dora-nexus").await;
    window.set_background_color(Color::new(0.12, 0.12, 0.18, 1.0));

    let mut scene = SceneNode3d::empty();
    scene.add_directional_light(Vec3::new(-1.0, -1.0, -1.0));
    scene.add_directional_light(Vec3::new(1.0, 1.0, 1.0));
    scene.add_directional_light(Vec3::new(0.0, 1.0, 0.0));

    let mut link_nodes = build_urdf_scene(&robot, urdf_dir, &mut scene);

    // Apply joint angles
    if !config.joints.is_empty() {
        set_joint_angles(&robot, &mut link_nodes, &config.joints);
        println!("[dora-nexus] Applied {} joint angles", config.joints.len());
    }

    let mut camera = OrbitCamera3d::new(config.camera_pos, config.camera_target);
    let mut frame: u64 = 0;
    let auto_exit = config.screenshot_path.is_some();

    while window.render_3d(&mut scene, &mut camera).await {
        // Draw world axes
        let o = Vec3::ZERO;
        window.draw_line(o, Vec3::new(0.15, 0.0, 0.0), Color::new(1.0, 0.0, 0.0, 1.0), 2.0, false);
        window.draw_line(o, Vec3::new(0.0, 0.15, 0.0), Color::new(0.0, 1.0, 0.0, 1.0), 2.0, false);
        window.draw_line(o, Vec3::new(0.0, 0.0, 0.15), Color::new(0.0, 0.0, 1.0, 1.0), 2.0, false);

        // Screenshot after a few frames (let GPU settle)
        if frame == 5 {
            if let Some(ref path) = config.screenshot_path {
                let image = window.snap_image();
                image.save(path).expect("Failed to save screenshot");
                println!("[dora-nexus] Screenshot saved: {}", path);
                if auto_exit {
                    break;
                }
            }
        }

        frame += 1;
    }
}
