"""Analyze screw trajectories: verify Z heights, headings, rotation, safe return."""

import json
import glob
import base64
import sys
import numpy as np
import torch
from pathlib import Path


def decode_joints(cmd, n_joints=7):
    """Decode joint positions from a trajectory command's CAN frames."""
    positions = []
    for fr in cmd["frames"][:n_joints]:
        raw = base64.b64decode(fr["data"])
        payload = raw[8:16]
        p_int = (payload[0] << 8) | payload[1]
        p = -12.5 + (p_int / 65535.0) * 25.0
        positions.append(p)
    return np.array(positions, dtype=np.float64)


def fk_position(chain, joints):
    """Get XYZ position from joint config."""
    q = torch.tensor([joints], dtype=torch.float32)
    with torch.no_grad():
        fk = chain.forward_kinematics(q)
        return fk.get_matrix()[0, :3, 3].numpy()


def fk_rotation(chain, joints):
    """Get rotation matrix from joint config."""
    q = torch.tensor([joints], dtype=torch.float32)
    with torch.no_grad():
        fk = chain.forward_kinematics(q)
        return fk.get_matrix()[0, :3, :3].numpy()


def analyze_trajectory(path, chain, label):
    """Analyze a single screw trajectory."""
    with open(path) as f:
        doc = json.load(f)
    meta = doc["metadata"]
    cmds = doc["commands"]
    n = len(cmds)
    arm = meta.get("arm", "?")
    gds = meta.get("grasp_dwell_start", n // 2)

    print(f"\n{'='*60}")
    print(f"{label}: {Path(path).name}")
    print(f"  arm={arm}, waypoints={n}, grasp_dwell_start={gds}")
    print(f"{'='*60}")

    # Sample Z profile through entire trajectory
    z_values = []
    xy_values = []
    sample_step = max(1, n // 40)
    for idx in range(0, n, sample_step):
        joints = decode_joints(cmds[idx])
        pos = fk_position(chain, joints)
        z_values.append((idx, pos[2]))
        xy_values.append((idx, pos[0], pos[1]))

    # Also get exact grasp point
    if gds < n:
        grasp_joints = decode_joints(cmds[gds])
        grasp_pos = fk_position(chain, grasp_joints)
        grasp_rot = fk_rotation(chain, grasp_joints)
        z_ee = grasp_rot[:, 2]  # approach direction
        y_ee = grasp_rot[:, 1]  # jaw axis
        heading = np.degrees(np.arctan2(z_ee[1], z_ee[0]))

        print(f"\n  GRASP POSE (wp {gds}):")
        print(f"    XYZ = [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]")
        print(f"    Z_ee (approach) = [{z_ee[0]:.3f}, {z_ee[1]:.3f}, {z_ee[2]:.3f}]")
        print(f"    Y_ee (jaw axis) = [{y_ee[0]:.3f}, {y_ee[1]:.3f}, {y_ee[2]:.3f}]")
        print(f"    Heading = {heading:.0f}°")

    # Z profile
    print(f"\n  Z PROFILE:")
    z_min = min(z for _, z in z_values)
    z_max = max(z for _, z in z_values)
    z_at_grasp = z_values[0][1]
    for idx, z in z_values:
        if gds < n:
            z_at_grasp = grasp_pos[2]
        marker = ""
        if abs(idx - gds) < sample_step:
            marker = " <-- GRASP"
        elif idx == 0:
            marker = " <-- START"
        elif idx >= n - sample_step:
            marker = " <-- END"
        elif z == z_min:
            marker = " <-- MIN"
        elif z == z_max:
            marker = " <-- MAX"
        print(f"    wp {idx:5d}: Z={z:.4f}{marker}")

    print(f"  Z range: {z_min:.4f} - {z_max:.4f} (span={z_max-z_min:.4f}m)")
    if gds < n:
        print(f"  Grasp Z: {grasp_pos[2]:.4f}")

    # Check for dangerous Z dips (below 0.05m = near table)
    # Exclude first/last 15% of trajectory (home position is naturally low)
    margin = max(10, n // 7)
    dangerous = [(idx, z) for idx, z in z_values if z < 0.05 and idx > margin and idx < n - margin]
    if dangerous:
        print(f"  WARNING: {len(dangerous)} waypoints below 50mm!")
        for idx, z in dangerous[:5]:
            print(f"    wp {idx}: Z={z:.4f}")

    # Check Z stability during hold/rotation phase
    if "hold" in label.lower():
        hold_start = gds + 2
        hold_zs = [z for idx, z in z_values if idx >= hold_start]
        if hold_zs:
            z_drift = max(hold_zs) - min(hold_zs)
            print(f"  Hold Z drift: {z_drift*1000:.1f}mm (should be <5mm)")
            if z_drift > 0.005:
                print(f"  WARNING: Hold Z drifts by {z_drift*1000:.1f}mm!")

    if "rotation" in label.lower() or "screw" in label.lower():
        # Check rotation phase Z stability
        rot_start = gds + 2
        # Find where rotation ends (before retract)
        rot_zs = []
        for idx, z in z_values:
            if idx >= rot_start and idx < n - n // 5:
                rot_zs.append(z)
        if rot_zs:
            z_drift = max(rot_zs) - min(rot_zs)
            print(f"  Rotation Z drift: {z_drift*1000:.1f}mm (should be <10mm)")

        # Check rotation direction by looking at yaw change
        if gds + 5 < n:
            rot0 = fk_rotation(chain, decode_joints(cmds[gds]))
            rot5 = fk_rotation(chain, decode_joints(cmds[min(gds + 20, n - 1)]))
            from scipy.spatial.transform import Rotation
            r_diff = Rotation.from_matrix(rot5 @ rot0.T)
            angle = r_diff.as_rotvec()
            yaw_change = np.degrees(angle[2])
            print(f"  Initial yaw change (first 20 wp): {yaw_change:.1f}°")
            # Clockwise from above = negative yaw = screw-in (tightening)
            if yaw_change > 0:
                print(f"  WARNING: Rotating counterclockwise (positive yaw) — should be clockwise for screw-in")
            else:
                print(f"  OK: Rotating clockwise (screw-in direction)")

    # Gripper actions
    ga = meta.get("gripper_actions", "[]")
    if isinstance(ga, str):
        ga = json.loads(ga)
    print(f"\n  GRIPPER ACTIONS ({len(ga)}):")
    for g in ga[:10]:
        rad = g["rad"]
        state = "OPEN" if rad < -0.5 else "CLOSED" if rad > -0.1 else f"partial({rad:.2f})"
        print(f"    wp {g['waypoint']:5d}: {state} (rad={rad:.3f})")
    if len(ga) > 10:
        print(f"    ... and {len(ga) - 10} more")


def main():
    traj_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    urdf_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Find URDF
    if urdf_path is None:
        for candidate in ["../openarm/openarm_v10.urdf", "examples/openarm/openarm_v10.urdf"]:
            if Path(candidate).exists():
                urdf_path = candidate
                break
    if urdf_path is None:
        print("ERROR: URDF not found. Pass as second argument.")
        sys.exit(1)

    # Import chain builder
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dora_motion_planner.main import build_chain

    # Find latest screw trajectories
    patterns = {
        "HOLD": "*screw_hold*.json",
        "SCREW ROTATION": "*screw_rotation*.json",
        "RELEASE": "*screw_release*.json",
    }

    found = False
    hold_z = None

    for label, pattern in patterns.items():
        files = sorted(glob.glob(str(Path(traj_dir) / pattern)))
        if not files:
            print(f"\n{label}: not found")
            continue
        found = True
        path = files[-1]

        with open(path) as f:
            doc = json.load(f)
        arm = doc["metadata"].get("arm", "left")
        ee = "openarm_left_hand_tcp" if arm == "left" else "openarm_right_hand_tcp"
        chain = build_chain(urdf_path, ee).to(dtype=torch.float32)

        analyze_trajectory(path, chain, label)

        # Track hold Z for comparison
        if "hold" in label.lower():
            gds = doc["metadata"].get("grasp_dwell_start", 0)
            cmds = doc["commands"]
            if gds < len(cmds):
                hold_z = fk_position(chain, decode_joints(cmds[gds]))[2]

    if not found:
        print(f"No screw trajectories found in {traj_dir}")
        sys.exit(1)

    # Summary
    if hold_z is not None:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"  Hold grasp Z: {hold_z:.4f}m")
        # Find screw Z
        screw_files = sorted(glob.glob(str(Path(traj_dir) / "*screw_rotation*.json")))
        if screw_files:
            with open(screw_files[-1]) as f:
                sdoc = json.load(f)
            sgds = sdoc["metadata"].get("grasp_dwell_start", 0)
            scmds = sdoc["commands"]
            sarm = sdoc["metadata"].get("arm", "left")
            see = "openarm_left_hand_tcp" if sarm == "left" else "openarm_right_hand_tcp"
            schain = build_chain(urdf_path, see).to(dtype=torch.float32)
            # If grasp_dwell_start is missing/0, estimate grasp as ~60% through trajectory
            if sgds == 0:
                sgds = len(scmds) * 3 // 5
            if sgds < len(scmds):
                screw_z = fk_position(schain, decode_joints(scmds[sgds]))[2]
                diff = screw_z - hold_z
                print(f"  Screw grasp Z: {screw_z:.4f}m")
                print(f"  Z difference: {diff*1000:.1f}mm (screw {'above' if diff > 0 else 'BELOW'} hold)")
                if diff < 0.01:
                    print(f"  WARNING: Screw arm should be higher than hold arm!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
