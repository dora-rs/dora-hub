"""Test trajectory smoothness: check velocity and acceleration profiles."""
import numpy as np
import sys
sys.path.insert(0, ".")
from demo_vertical_lift import compute_trajectory, fk

traj = compute_trajectory(60, 0.12)
dt = 0.1  # 10Hz

# Velocity (first derivative)
vel = np.diff(traj, axis=0) / dt
# Acceleration (second derivative)
acc = np.diff(vel, axis=0) / dt
# Jerk (third derivative)
jerk = np.diff(acc, axis=0) / dt

print("=== Per-joint max velocity (rad/s) ===")
print(np.max(np.abs(vel), axis=0))

print("\n=== Per-joint max acceleration (rad/s²) ===")
print(np.max(np.abs(acc), axis=0))

print("\n=== Per-joint max jerk (rad/s³) ===")
print(np.max(np.abs(jerk), axis=0))

# Find the steps with highest acceleration
acc_norm = np.linalg.norm(acc, axis=1)
worst_steps = np.argsort(acc_norm)[-5:][::-1]
print(f"\n=== Top 5 acceleration spikes (step, acc_norm) ===")
for s in worst_steps:
    print(f"  step {s}: acc_norm={acc_norm[s]:.3f} rad/s²  joints={acc[s]}")

# Also check EE position smoothness
ee_pos = np.array([fk(q)[:3, 3] for q in traj])
ee_vel = np.diff(ee_pos, axis=0) / dt
ee_acc = np.diff(ee_vel, axis=0) / dt

print(f"\n=== EE position max velocity (m/s) ===")
print(f"  dx={np.max(np.abs(ee_vel[:, 0])):.6f}  dy={np.max(np.abs(ee_vel[:, 1])):.6f}  dz={np.max(np.abs(ee_vel[:, 2])):.6f}")

print(f"\n=== EE position max acceleration (m/s²) ===")
print(f"  ddx={np.max(np.abs(ee_acc[:, 0])):.6f}  ddy={np.max(np.abs(ee_acc[:, 1])):.6f}  ddz={np.max(np.abs(ee_acc[:, 2])):.6f}")

# Print full velocity profile for J1 and J4 (the main movers)
print("\n=== J1 velocity profile (rad/s) ===")
for i, v in enumerate(vel[:, 0]):
    bar = "#" * int(abs(v) * 20)
    print(f"  step {i:3d}: {v:+.4f}  {bar}")

print("\n=== J4 velocity profile (rad/s) ===")
for i, v in enumerate(vel[:, 3]):
    bar = "#" * int(abs(v) * 10)
    print(f"  step {i:3d}: {v:+.4f}  {bar}")
