"""Trajectory analysis — discontinuities, velocity, acceleration, joint limits, smoothness.

Library + CLI.  All analysis functions take ``(T, 7)`` float32 numpy arrays
and return structured dicts.  The motion planner calls these live after
planning; the CLI allows offline analysis of saved JSON files.

CLI usage::

    python -m dora_motion_planner.analyze_trajectory trajectories/*.json [--plot] [--max-step 0.1]

Or via the installed entry point::

    analyze-trajectory trajectories/*.json [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# OpenArm joint limits (radians) — 7 joints, from openarm_v10.urdf.
# J0 and J1 are mirrored between left/right arms; J2-J6 are the same.
OPENARM_JOINT_LIMITS = {
    "left": (
        np.array([-3.490659, -3.316125, -1.570796, 0.0, -1.570796, -0.785398, -1.570796], dtype=np.float32),
        np.array([1.396263, 0.174533, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796], dtype=np.float32),
    ),
    "right": (
        np.array([-1.396263, -0.174533, -1.570796, 0.0, -1.570796, -0.785398, -1.570796], dtype=np.float32),
        np.array([3.490659, 3.316125, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796], dtype=np.float32),
    ),
}


def analyze_discontinuities(
    traj: np.ndarray, max_step: float = 0.1
) -> list[dict]:
    """Find waypoints where any joint jumps more than *max_step* radians."""
    diffs = np.abs(np.diff(traj, axis=0))  # (T-1, J)
    results = []
    for t in range(diffs.shape[0]):
        for j in range(diffs.shape[1]):
            if diffs[t, j] > max_step:
                results.append({
                    "waypoint": int(t + 1),
                    "joint": int(j),
                    "delta_rad": float(diffs[t, j]),
                    "delta_deg": float(np.degrees(diffs[t, j])),
                })
    return results


def analyze_velocity_acceleration(
    traj: np.ndarray, dt: float
) -> dict:
    """Per-joint max/mean velocity and max/rms acceleration, with spike flags."""
    vel = np.diff(traj, axis=0) / dt  # (T-1, J) rad/s
    acc = np.diff(vel, axis=0) / dt   # (T-2, J) rad/s^2

    num_joints = traj.shape[1]
    result = {"joints": [], "velocity": vel, "acceleration": acc}

    for j in range(num_joints):
        v = np.abs(vel[:, j])
        a = np.abs(acc[:, j]) if acc.shape[0] > 0 else np.array([0.0])
        median_v = float(np.median(v)) if len(v) > 0 else 0.0
        median_a = float(np.median(a)) if len(a) > 0 else 0.0

        joint_info = {
            "max_velocity": float(np.max(v)) if len(v) > 0 else 0.0,
            "mean_velocity": float(np.mean(v)) if len(v) > 0 else 0.0,
            "max_acceleration": float(np.max(a)) if len(a) > 0 else 0.0,
            "rms_acceleration": float(np.sqrt(np.mean(a**2))) if len(a) > 0 else 0.0,
            "velocity_spikes": int(np.sum(v > 5 * median_v)) if median_v > 1e-6 else 0,
            "acceleration_spikes": int(np.sum(a > 5 * median_a)) if median_a > 1e-6 else 0,
        }
        result["joints"].append(joint_info)

    return result


def analyze_joint_limits(
    traj: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    arm: str = "left",
) -> list[dict]:
    """Check trajectory against joint limits, return violations."""
    if lower is None or upper is None:
        defaults = OPENARM_JOINT_LIMITS.get(arm, OPENARM_JOINT_LIMITS["left"])
        if lower is None:
            lower = defaults[0][: traj.shape[1]]
        if upper is None:
            upper = defaults[1][: traj.shape[1]]

    violations = []
    for j in range(traj.shape[1]):
        below = np.where(traj[:, j] < lower[j])[0]
        above = np.where(traj[:, j] > upper[j])[0]
        for t in below:
            violations.append({
                "waypoint": int(t),
                "joint": int(j),
                "value": float(traj[t, j]),
                "limit": float(lower[j]),
                "type": "below_lower",
            })
        for t in above:
            violations.append({
                "waypoint": int(t),
                "joint": int(j),
                "value": float(traj[t, j]),
                "limit": float(upper[j]),
                "type": "above_upper",
            })
    return violations


def analyze_smoothness(traj: np.ndarray, dt: float) -> dict:
    """Jerk (derivative of acceleration) and total joint path length."""
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt  # (T-3, J)

    path_length = np.sum(np.abs(np.diff(traj, axis=0)), axis=0)  # per joint

    return {
        "max_jerk": [float(np.max(np.abs(jerk[:, j]))) if jerk.shape[0] > 0 else 0.0
                     for j in range(traj.shape[1])],
        "rms_jerk": [float(np.sqrt(np.mean(jerk[:, j] ** 2))) if jerk.shape[0] > 0 else 0.0
                     for j in range(traj.shape[1])],
        "path_length_rad": [float(path_length[j]) for j in range(traj.shape[1])],
        "total_path_length": float(np.sum(path_length)),
    }


def summarize(
    traj: np.ndarray,
    dt: float,
    metadata: dict | None = None,
    max_step: float = 0.1,
) -> dict:
    """Run all analyses and return a combined results dict with quality rating."""
    arm = (metadata or {}).get("arm", "left")
    discontinuities = analyze_discontinuities(traj, max_step)
    vel_acc = analyze_velocity_acceleration(traj, dt)
    joint_limits = analyze_joint_limits(traj, arm=arm)
    smoothness = analyze_smoothness(traj, dt)

    # Quality rating
    quality = "OK"
    reasons = []
    if discontinuities:
        quality = "WARN"
        reasons.append("discontinuities")
    if joint_limits:
        quality = "FAIL"
        reasons.append("joint_limit_violations")
    # Flag if any joint has velocity spikes
    if any(j["velocity_spikes"] > 0 for j in vel_acc["joints"]):
        if quality == "OK":
            quality = "WARN"
        reasons.append("velocity_spikes")

    # Strip large arrays from vel_acc for the summary (keep per-joint stats)
    vel_acc_summary = {"joints": vel_acc["joints"]}

    return {
        "num_waypoints": traj.shape[0],
        "num_joints": traj.shape[1],
        "duration": round(traj.shape[0] * dt, 2),
        "dt": dt,
        "metadata": metadata or {},
        "discontinuities": discontinuities,
        "velocity_acceleration": vel_acc_summary,
        "joint_limits": joint_limits,
        "smoothness": smoothness,
        "quality": quality,
        "quality_reasons": reasons,
    }


def format_report(path_or_name: str, results: dict) -> str:
    """Format a multi-line summary string for printing."""
    lines = []
    lines.append(f"=== Trajectory Analysis: {path_or_name} ===")

    arm = results.get("metadata", {}).get("arm", "?")
    lines.append(
        f"Waypoints: {results['num_waypoints']} | "
        f"Duration: {results['duration']}s | "
        f"Arm: {arm}"
    )

    # Discontinuities
    discs = results["discontinuities"]
    if discs:
        lines.append(f"Discontinuities: {len(discs)} found")
        for d in discs[:10]:  # show at most 10
            lines.append(
                f"  t={d['waypoint']} joint {d['joint']}: "
                f"{d['delta_rad']:.3f} rad ({d['delta_deg']:.1f} deg)"
            )
        if len(discs) > 10:
            lines.append(f"  ... and {len(discs) - 10} more")
    else:
        lines.append("Discontinuities: none")

    # Velocity / acceleration
    va = results["velocity_acceleration"]
    max_v = " ".join(
        f"J{j}={info['max_velocity']:.1f}"
        for j, info in enumerate(va["joints"])
    )
    max_a = " ".join(
        f"J{j}={info['max_acceleration']:.0f}"
        for j, info in enumerate(va["joints"])
    )
    lines.append(f"Max velocity: {max_v} rad/s")
    lines.append(f"Max accel: {max_a} rad/s^2")

    # Joint limit violations
    jl = results["joint_limits"]
    if jl:
        lines.append(f"Joint limit violations: {len(jl)}")
        for v in jl[:5]:
            lines.append(
                f"  t={v['waypoint']} joint {v['joint']}: "
                f"{v['value']:.3f} rad ({v['type']}, limit={v['limit']:.3f})"
            )
    else:
        lines.append("Joint limits: OK")

    # Smoothness
    sm = results["smoothness"]
    lines.append(f"Total path length: {sm['total_path_length']:.2f} rad")

    # Quality
    q = results["quality"]
    reasons = ", ".join(results["quality_reasons"]) if results["quality_reasons"] else ""
    if reasons:
        lines.append(f"Quality: {q} ({reasons})")
    else:
        lines.append(f"Quality: {q}")

    return "\n".join(lines)


def plot_trajectory(traj: np.ndarray, dt: float, results: dict, out_path: str):
    """Plot joint angles, velocities, and accelerations as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_joints = traj.shape[1]
    t_pos = np.arange(traj.shape[0]) * dt
    t_vel = t_pos[:-1] + dt / 2
    t_acc = t_pos[:-2] + dt

    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Discontinuity waypoints for marking
    disc_waypoints = {d["waypoint"] for d in results["discontinuities"]}

    # Joint angles
    ax = axes[0]
    for j in range(num_joints):
        ax.plot(t_pos, traj[:, j], label=f"J{j}", linewidth=0.8)
    # Mark discontinuities
    for wp in disc_waypoints:
        if wp < len(t_pos):
            ax.axvline(t_pos[wp], color="red", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Position (rad)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Joint Angles")
    ax.grid(True, alpha=0.3)

    # Velocities
    ax = axes[1]
    for j in range(num_joints):
        ax.plot(t_vel, vel[:, j], label=f"J{j}", linewidth=0.8)
    for wp in disc_waypoints:
        if wp < len(t_vel):
            ax.axvline(t_vel[wp], color="red", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Joint Velocities")
    ax.grid(True, alpha=0.3)

    # Accelerations
    ax = axes[2]
    for j in range(num_joints):
        ax.plot(t_acc, acc[:, j], label=f"J{j}", linewidth=0.8)
    for wp in disc_waypoints:
        if wp - 1 < len(t_acc) and wp - 1 >= 0:
            ax.axvline(t_acc[wp - 1], color="red", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Acceleration (rad/s²)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Joint Accelerations")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main():
    """CLI entry point for offline trajectory analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze OpenArm trajectory JSON files",
    )
    parser.add_argument("files", nargs="+", help="Trajectory JSON files")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.1,
        help="Discontinuity threshold in radians (default: 0.1)",
    )
    args = parser.parse_args()

    from dora_motion_planner.trajectory_json import load

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        traj, metadata = load(path)
        dt = float(metadata.get("dt", 0.1))
        results = summarize(traj, dt, metadata=metadata, max_step=args.max_step)
        print(format_report(path.name, results))
        print()

        if args.plot:
            plot_path = path.with_suffix(".png")
            plot_trajectory(traj, dt, results, str(plot_path))


if __name__ == "__main__":
    main()
