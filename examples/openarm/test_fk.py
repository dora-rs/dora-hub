"""Find J7 that keeps gripper pointing down for each J1,J4 combo."""
import numpy as np

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
def trans(x, y, z):
    T = np.eye(4); T[0,3], T[1,3], T[2,3] = x, y, z; return T
def rpy_to_rot(r, p, y):
    return rot_z(y) @ rot_y(p) @ rot_x(r)

def fk_left_arm(q):
    T = trans(0, 0.031, 0.698) @ rpy_to_rot(-1.5708, 0, 0)
    T = T @ trans(0, 0, 0.0625) @ rot_z(q[0])
    T = T @ trans(-0.0301, 0, 0.06) @ rpy_to_rot(-np.pi/2, 0, 0) @ rot_x(-q[1])
    T = T @ trans(0.0301, 0, 0.06625) @ rot_z(q[2])
    T = T @ trans(0, 0.0315, 0.15375) @ rot_y(q[3])
    T = T @ trans(0, -0.0315, 0.0955) @ rot_z(q[4])
    T = T @ trans(0.0375, 0, 0.1205) @ rot_x(q[5])
    T = T @ trans(-0.0375, 0, 0) @ rot_y(-q[6])
    T = T @ trans(1e-6, 0.0205, 0)
    return T

q0 = [0]*7
p0 = fk_left_arm(q0)[:3, 3]

# For each t along trajectory, find J7 that minimizes |EE_z - (0,0,-1)|
print("Optimal J7 for gripper-down at each trajectory point:")
print(f"{'t':>4} | {'J1':>6} {'J4':>6} {'J7_opt':>8} | {'dx':>7} {'dy':>7} {'dz':>7} | {'Zx':>6} {'Zy':>6} {'Zz':>6}")

j7_values = []
for i in range(11):
    t = i / 10.0
    j1 = 0.75 * t
    j4 = 1.5 * t

    # Search J7 and also J5, J6 for best gripper-down
    best_j7 = 0
    best_score = 1e9
    for j7_i in range(-314, 315):
        j7 = j7_i * 0.005
        if j7 < -1.57 or j7 > 1.57:
            continue
        q = [j1, 0, 0, j4, 0, 0, j7]
        T = fk_left_arm(q)
        z = T[:3, 2]
        # Want z close to (0, 0, -1)
        score = (z[0])**2 + (z[1])**2 + (z[2] + 1)**2
        if score < best_score:
            best_score = score
            best_j7 = j7

    q = [j1, 0, 0, j4, 0, 0, best_j7]
    T = fk_left_arm(q)
    p, z = T[:3,3], T[:3,2]
    dx, dy, dz = p[0]-p0[0], p[1]-p0[1], p[2]-p0[2]
    j7_values.append(best_j7)
    print(f"{t:.1f} | {j1:>6.3f} {j4:>6.3f} {best_j7:>8.3f} | {dx:>+7.4f} {dy:>+7.4f} {dz:>+7.4f} | {z[0]:>6.3f} {z[1]:>6.3f} {z[2]:>6.3f}")

# Check if J7 ≈ k * J4 for some constant k
print(f"\nJ7/J4 ratios:")
for i in range(1, 11):
    t = i / 10.0
    j4 = 1.5 * t
    ratio = j7_values[i] / j4 if j4 != 0 else 0
    print(f"  t={t:.1f}: J4={j4:.3f} J7={j7_values[i]:.3f} ratio={ratio:.4f}")

# Also try with J6 for compensation
print("\n\nSearching J6+J7 for best gripper-down at t=1.0:")
j1, j4 = 0.75, 1.5
best = None
best_score = 1e9
for j6_i in range(-157, 158):
    j6 = j6_i * 0.01
    if j6 < -0.785 or j6 > 0.785:
        continue
    for j7_i in range(-157, 158):
        j7 = j7_i * 0.01
        if j7 < -1.57 or j7 > 1.57:
            continue
        q = [j1, 0, 0, j4, 0, j6, j7]
        T = fk_left_arm(q)
        z = T[:3, 2]
        score = (z[0])**2 + (z[1])**2 + (z[2] + 1)**2
        if score < best_score:
            best_score = score
            best = (j6, j7, z)

if best:
    j6, j7, z = best
    print(f"Best: J6={j6:.3f}, J7={j7:.3f} -> Z=({z[0]:.4f}, {z[1]:.4f}, {z[2]:.4f})")
