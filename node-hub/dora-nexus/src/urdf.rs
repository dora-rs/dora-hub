use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use kiss3d::glamx::{EulerRot, Quat, Vec3};
use kiss3d::color::Color;
use kiss3d::procedural::{IndexBuffer, RenderMesh};
use kiss3d::scene::SceneNode3d;

/// Load an STL file and convert it to a kiss3d RenderMesh.
pub fn load_stl(path: &Path) -> Option<RenderMesh> {
    let mut file = File::open(path).ok()?;
    let mesh = stl_io::read_stl(&mut file).ok()?;

    let coords: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|v| Vec3::new(v[0], v[1], v[2]))
        .collect();

    let normals: Vec<Vec3> = {
        let mut vnormals = vec![Vec3::ZERO; coords.len()];
        for face in &mesh.faces {
            let n = Vec3::new(face.normal[0], face.normal[1], face.normal[2]);
            for &vi in &face.vertices {
                vnormals[vi] += n;
            }
        }
        vnormals
            .iter()
            .map(|n| {
                let len = n.length();
                if len > 0.0 {
                    *n / len
                } else {
                    Vec3::Y
                }
            })
            .collect()
    };

    let indices: Vec<[u32; 3]> = mesh
        .faces
        .iter()
        .map(|f| {
            [
                f.vertices[0] as u32,
                f.vertices[1] as u32,
                f.vertices[2] as u32,
            ]
        })
        .collect();

    Some(RenderMesh::new(
        coords,
        Some(normals),
        None,
        Some(IndexBuffer::Unified(indices)),
    ))
}

/// Quaternion from URDF RPY (roll-pitch-yaw) convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
pub fn quat_from_rpy(rpy: &[f64; 3]) -> Quat {
    let roll = rpy[0] as f32;
    let pitch = rpy[1] as f32;
    let yaw = rpy[2] as f32;
    Quat::from_euler(EulerRot::ZYX, yaw, pitch, roll)
}

/// Resolve a mesh filename relative to the URDF directory.
pub fn resolve_mesh_path(urdf_dir: &Path, filename: &str) -> PathBuf {
    let cleaned = filename
        .trim_start_matches("package://")
        .trim_start_matches("file://")
        .trim_start_matches("./");
    urdf_dir.join(cleaned)
}

/// Build a scene graph from a URDF robot model.
/// Returns a map from link name to its scene node.
pub fn build_urdf_scene(
    robot: &urdf_rs::Robot,
    urdf_dir: &Path,
    scene: &mut SceneNode3d,
) -> HashMap<String, SceneNode3d> {
    let mut link_nodes: HashMap<String, SceneNode3d> = HashMap::new();

    for link in &robot.links {
        let mut node = SceneNode3d::empty();

        for visual in &link.visual {
            let origin_xyz = &visual.origin.xyz;
            let origin_rpy = &visual.origin.rpy;

            match &visual.geometry {
                urdf_rs::Geometry::Mesh { filename, scale } => {
                    let mesh_path = resolve_mesh_path(urdf_dir, filename);
                    if let Some(render_mesh) = load_stl(&mesh_path) {
                        let s = scale
                            .as_ref()
                            .map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
                            .unwrap_or(Vec3::ONE);

                        let mut mesh_node = node.add_render_mesh(render_mesh, s);
                        mesh_node.set_position(Vec3::new(
                            origin_xyz[0] as f32,
                            origin_xyz[1] as f32,
                            origin_xyz[2] as f32,
                        ));
                        mesh_node.set_rotation(quat_from_rpy(&[
                            origin_rpy[0], origin_rpy[1], origin_rpy[2],
                        ]));
                        mesh_node.set_color(Color::new(0.7, 0.7, 0.8, 1.0));
                    } else {
                        eprintln!("Failed to load mesh: {}", mesh_path.display());
                    }
                }
                urdf_rs::Geometry::Box { size } => {
                    let mut box_node = node.add_cube(
                        size[0] as f32,
                        size[1] as f32,
                        size[2] as f32,
                    );
                    box_node.set_position(Vec3::new(
                        origin_xyz[0] as f32,
                        origin_xyz[1] as f32,
                        origin_xyz[2] as f32,
                    ));
                    box_node.set_color(Color::new(0.7, 0.7, 0.8, 1.0));
                }
                urdf_rs::Geometry::Cylinder { radius, length } => {
                    let mut cyl_node = node.add_cylinder(*radius as f32, *length as f32);
                    cyl_node.set_position(Vec3::new(
                        origin_xyz[0] as f32,
                        origin_xyz[1] as f32,
                        origin_xyz[2] as f32,
                    ));
                    cyl_node.set_color(Color::new(0.7, 0.7, 0.8, 1.0));
                }
                urdf_rs::Geometry::Sphere { radius } => {
                    let mut sph_node = node.add_sphere(*radius as f32);
                    sph_node.set_position(Vec3::new(
                        origin_xyz[0] as f32,
                        origin_xyz[1] as f32,
                        origin_xyz[2] as f32,
                    ));
                    sph_node.set_color(Color::new(0.7, 0.7, 0.8, 1.0));
                }
                _ => {}
            }
        }

        link_nodes.insert(link.name.clone(), node);
    }

    let mut child_links: Vec<String> = Vec::new();

    for joint in &robot.joints {
        let parent_name = &joint.parent.link;
        let child_name = &joint.child.link;
        child_links.push(child_name.clone());

        if let Some(child_node) = link_nodes.get_mut(child_name) {
            let xyz = &joint.origin.xyz;
            let rpy = &joint.origin.rpy;
            child_node.set_position(Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32));
            child_node.set_rotation(quat_from_rpy(&[rpy[0], rpy[1], rpy[2]]));
        }

        if let Some(child_node) = link_nodes.remove(child_name) {
            if let Some(parent_node) = link_nodes.get_mut(parent_name) {
                parent_node.add_child(child_node.clone());
                link_nodes.insert(child_name.clone(), child_node);
            }
        }
    }

    for link in &robot.links {
        if !child_links.contains(&link.name) {
            if let Some(node) = link_nodes.get(&link.name) {
                scene.add_child(node.clone());
            }
        }
    }

    link_nodes
}

/// Apply joint angles to the URDF scene nodes.
pub fn set_joint_angles(
    robot: &urdf_rs::Robot,
    link_nodes: &mut HashMap<String, SceneNode3d>,
    joint_angles: &HashMap<String, f32>,
) {
    for joint in &robot.joints {
        let angle = joint_angles.get(&joint.name).copied().unwrap_or(0.0);
        if matches!(
            joint.joint_type,
            urdf_rs::JointType::Revolute | urdf_rs::JointType::Continuous
        ) {
            let axis = Vec3::new(
                joint.axis.xyz[0] as f32,
                joint.axis.xyz[1] as f32,
                joint.axis.xyz[2] as f32,
            );
            let xyz = &joint.origin.xyz;
            let rpy = &joint.origin.rpy;
            let origin_rot = quat_from_rpy(&[rpy[0], rpy[1], rpy[2]]);
            let joint_rot = Quat::from_axis_angle(axis, angle);
            let combined = origin_rot * joint_rot;

            if let Some(node) = link_nodes.get_mut(&joint.child.link) {
                node.set_position(Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32));
                node.set_rotation(combined);
            }
        } else if matches!(joint.joint_type, urdf_rs::JointType::Prismatic) {
            let axis = Vec3::new(
                joint.axis.xyz[0] as f32,
                joint.axis.xyz[1] as f32,
                joint.axis.xyz[2] as f32,
            );
            let xyz = &joint.origin.xyz;
            let rpy = &joint.origin.rpy;
            let origin_pos = Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32);
            let translation = origin_pos + axis * angle;

            if let Some(node) = link_nodes.get_mut(&joint.child.link) {
                node.set_position(translation);
                node.set_rotation(quat_from_rpy(&[rpy[0], rpy[1], rpy[2]]));
            }
        }
    }
}

/// Clean URDF string by replacing empty <geometry> tags with a tiny dummy box.
pub fn clean_urdf_string(urdf_string: &str) -> String {
    let mut urdf_clean = String::new();
    let mut chars = urdf_string;
    while let Some(start) = chars.find("<geometry>") {
        urdf_clean.push_str(&chars[..start]);
        let after_tag = &chars[start + "<geometry>".len()..];
        if let Some(end) = after_tag.find("</geometry>") {
            let inner = &after_tag[..end];
            if inner.trim().is_empty() {
                urdf_clean.push_str("<geometry><box size=\"0.001 0.001 0.001\"/></geometry>");
            } else {
                urdf_clean.push_str("<geometry>");
                urdf_clean.push_str(inner);
                urdf_clean.push_str("</geometry>");
            }
            chars = &after_tag[end + "</geometry>".len()..];
        } else {
            urdf_clean.push_str("<geometry>");
            chars = after_tag;
        }
    }
    urdf_clean.push_str(chars);
    urdf_clean
}
