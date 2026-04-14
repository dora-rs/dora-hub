#!/usr/bin/env python3
"""Benchmark: TRAC-IK vs Pinocchio vs Adam IK for OpenArm.

Compares all three solvers across workspace targets on accuracy,
speed, success rate, and trajectory quality (via analyze_trajectory).

Usage (on baguette):
    cd ~/dora-hub
    python3 node-hub/dora-motion-planner/tests/benchmark_ik_solvers.py
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import pinocchio as pin
from trac_ik import trac_ik
import torch
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET

from dora_motion_planner.analyze_trajectory import (
    summarize,
    format_report,
    OPENARM_JOINT_LIMITS,
)
from dora_motion_planner.main import _ik_batch_adam

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
URDF_PATH = str(Path(__file__).resolve().parents[3] / "examples" / "openarm" / "openarm_v10.urdf")
EE_LINK = "openarm_left_hand_tcp"
ARM = "left"
NUM_RANDOM = 200
POS_THRESHOLD = 0.005  # 5mm
DEVICE = "cpu"  # baguette has no GPU for torch


def build_chain(urdf_path, ee_link):
    with open(urdf_path) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    clean_urdf = ET.tostring(root, encoding="unicode")
    return pk.build_serial_chain_from_urdf(clean_urdf, ee_link)


def fk_pk(chain, q):
    with torch.no_grad():
        qt = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        fk = chain.forward_kinematics(qt)
        mat = fk.get_matrix()[0]
        return mat[:3, 3].numpy(), mat[:3, :3].numpy()


def clean_urdf_for_tracik(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(ET.tostring(root, encoding="unicode"))
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
class PinIK:
    def __init__(self, model, data, frame_id, joint_indices):
        self.model, self.data = model, data
        self.frame_id, self.joint_indices = frame_id, joint_indices

    def solve(self, target_pos, q_init=None, max_iter=200, eps=1e-3, damp=1e-6,
              target_rot=None, rot_weight=0.5, nullspace_weight=1.0):
        q = pin.neutral(self.model)
        q_ref = q.copy()
        if q_init is not None:
            for i, ji in enumerate(self.joint_indices):
                q[ji] = float(q_init[i])
                q_ref[ji] = float(q_init[i])
        use_rot = target_rot is not None
        nv = self.model.nv
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.frame_id]
            pos_err = target_pos - oMf.translation
            if use_rot:
                rot_err = pin.log3(target_rot @ oMf.rotation.T)
                err = np.concatenate([pos_err, rot_weight * rot_err])
                if np.linalg.norm(pos_err) < eps and np.linalg.norm(rot_err) < eps * 5:
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
                J[3:, :] *= rot_weight
            else:
                err = pos_err
                if np.linalg.norm(err) < eps:
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            JJt = J @ J.T + damp * np.eye(J.shape[0])
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
            dq = J_pinv @ err
            if nullspace_weight > 0 and q_init is not None:
                N = np.eye(nv) - J_pinv @ J
                dq += N @ (nullspace_weight * (q_ref - q))
            q = pin.integrate(self.model, q, dq)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        final_err = np.linalg.norm(target_pos - self.data.oMf[self.frame_id].translation)
        q_arm = np.array([q[ji] for ji in self.joint_indices], dtype=np.float32)
        return q_arm, final_err


def build_pin_ik(urdf_path, ee_link, arm_side):
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    frame_id = model.getFrameId(ee_link)
    prefix = "L_" if arm_side == "left" else "R_"
    joint_indices = []
    for i in range(1, model.njoints):
        name = model.names[i]
        if name.startswith(prefix) and name.endswith(tuple(f"J{j}" for j in range(1, 8))):
            joint_indices.append(i - 1)
    return PinIK(model, data, frame_id, joint_indices)


class TracIKSolver:
    def __init__(self, urdf_path, base_link, ee_link, timeout=0.005, epsilon=1e-5,
                 solver_type="Speed"):
        clean_path = clean_urdf_for_tracik(urdf_path)
        self.solver = trac_ik.TracIK(base_link, ee_link, clean_path,
                                      timeout=timeout, epsilon=epsilon,
                                      solver_type=solver_type)
        self.dof = self.solver.dof

    def solve(self, target_pos, target_rot, q_init=None):
        if q_init is None:
            q_init = np.zeros(self.dof, dtype=np.float64)
        else:
            q_init = np.asarray(q_init, dtype=np.float64)
        result = self.solver.ik(
            np.asarray(target_pos, dtype=np.float64),
            np.asarray(target_rot, dtype=np.float64),
            q_init,
        )
        if result is None:
            return None, float("inf")
        q = np.asarray(result, dtype=np.float32)
        fk_pos, _ = self.solver.fk(q.astype(np.float64))
        return q, float(np.linalg.norm(fk_pos - target_pos))


class AdamIKSolver:
    def __init__(self, chain, device, arm="left"):
        self.ik_chain = chain.to(dtype=torch.float32, device=device)
        self.device = device
        limits = OPENARM_JOINT_LIMITS[arm]
        self.lower = torch.tensor(limits[0], dtype=torch.float32, device=device)
        self.upper = torch.tensor(limits[1], dtype=torch.float32, device=device)

    def solve(self, target_pos, target_rot, q_init, num_seeds=8, max_iters=2000,
              rot_weight=0.0):
        t_pos = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
        t_rot = None
        if target_rot is not None and rot_weight > 0:
            t_rot = torch.tensor(target_rot, dtype=torch.float32, device=self.device)
        if torch.is_tensor(q_init):
            c = q_init.to(self.device)
        else:
            c = torch.tensor(q_init, dtype=torch.float32, device=self.device)
        q, err = _ik_batch_adam(
            self.ik_chain, t_pos, t_rot, c,
            self.lower, self.upper, self.device,
            num_seeds=num_seeds, max_iters=max_iters, rot_weight=rot_weight,
        )
        return q.cpu().numpy() if q is not None else None, err


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------
def generate_fk_targets(chain, n, arm="left"):
    """Reachable targets via FK from random joint configs."""
    limits = OPENARM_JOINT_LIMITS[arm]
    lower, upper = limits
    targets = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        q = lower + (upper - lower) * rng.random(len(lower)).astype(np.float32)
        pos, rot = fk_pk(chain, q)
        targets.append({"pos": pos.copy(), "rot": rot.copy(), "q_true": q.copy()})
    return targets


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------
def build_trajectory(targets, solver_fn, q_init, num_interp=20):
    waypoints = [np.array(q_init, dtype=np.float32)]
    solve_times, errors, successes = [], [], []
    q_prev = np.array(q_init, dtype=np.float32)
    for tgt in targets:
        t0 = time.perf_counter()
        q, err = solver_fn(tgt["pos"], tgt["rot"], q_prev)
        solve_times.append(time.perf_counter() - t0)
        errors.append(err)
        if q is not None and err < POS_THRESHOLD:
            successes.append(True)
            waypoints.append(np.asarray(q, dtype=np.float32))
            q_prev = waypoints[-1]
        else:
            successes.append(False)
    if len(waypoints) < 2:
        return None, solve_times, errors, successes
    parts = []
    for i in range(len(waypoints) - 1):
        for t in range(num_interp):
            alpha = t / num_interp
            parts.append(waypoints[i] * (1 - alpha) + waypoints[i + 1] * alpha)
    parts.append(waypoints[-1])
    return np.array(parts, dtype=np.float32), solve_times, errors, successes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_err(e):
    if e == float("inf"):
        return "FAIL(inf)"
    elif e > 0.1:
        return f"FAIL({e*1000:.0f}mm)"
    elif e > POS_THRESHOLD:
        return f"POOR({e*1000:.1f}mm)"
    else:
        return f"OK({e*1000:.2f}mm)"


def run_test(name, targets, solvers, use_perturbed_seed=False, seed_perturbation=0.1):
    """Run a set of IK targets through multiple solvers and print results."""
    rng = np.random.default_rng(0)
    results = {}
    for sname, sfn in solvers:
        times, errs, successes = [], [], 0
        for tgt in targets:
            if use_perturbed_seed and "q_true" in tgt:
                q_init = tgt["q_true"] + rng.normal(0, seed_perturbation, 7).astype(np.float32)
            else:
                q_init = np.zeros(7, dtype=np.float32)
            t0 = time.perf_counter()
            q, err = sfn(tgt["pos"], tgt["rot"], q_init)
            times.append(time.perf_counter() - t0)
            errs.append(err if err != float("inf") else 999.0)
            if q is not None and err < POS_THRESHOLD:
                successes += 1
        ea = np.array(errs)
        good = ea[ea < POS_THRESHOLD]
        results[sname] = {
            "success": f"{successes}/{len(targets)}",
            "rate": successes / len(targets),
            "mean_ms": np.mean(times) * 1000,
            "med_ms": np.median(times) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
            "mean_err_mm": np.mean(good) * 1000 if len(good) > 0 else float("inf"),
            "med_err_mm": np.median(good) * 1000 if len(good) > 0 else float("inf"),
        }
    print(f"\n{'Solver':<22} {'Success':>9} {'Med(ms)':>9} {'P99(ms)':>9} {'MedErr':>10}")
    print("-" * 65)
    for sname, r in results.items():
        me = f"{r['med_err_mm']:.2f}mm" if r['med_err_mm'] != float("inf") else "inf"
        print(f"{sname:<22} {r['success']:>9} {r['med_ms']:>9.2f} {r['p99_ms']:>9.2f} {me:>10}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  IK Solver Benchmark: TRAC-IK vs Pinocchio vs Adam")
    print("=" * 70)
    print(f"URDF: {URDF_PATH}")
    print(f"Arm: {ARM}, EE: {EE_LINK}, Device: {DEVICE}")
    print()

    chain = build_chain(URDF_PATH, EE_LINK)
    pin_ik = build_pin_ik(URDF_PATH, EE_LINK, ARM)
    trac_speed = TracIKSolver(URDF_PATH, "world", EE_LINK, timeout=0.005, solver_type="Speed")
    trac_dist = TracIKSolver(URDF_PATH, "world", EE_LINK, timeout=0.005, solver_type="Distance")
    trac_long = TracIKSolver(URDF_PATH, "world", EE_LINK, timeout=0.05, solver_type="Speed")
    adam_ik = AdamIKSolver(chain, DEVICE, ARM)

    print(f"  PinIK: {len(pin_ik.joint_indices)}J | TracIK: {trac_speed.dof}J | Adam: GPU={DEVICE}")
    print()

    # Solver function signatures: (pos, rot, q_init) -> (q, err)
    def pin_pos(pos, rot, qi):
        return pin_ik.solve(np.float64(pos), q_init=qi)

    def pin_6dof(pos, rot, qi):
        return pin_ik.solve(np.float64(pos), q_init=qi, target_rot=np.float64(rot), rot_weight=0.5)

    def trac_s(pos, rot, qi):
        return trac_speed.solve(pos, rot, q_init=qi)

    def trac_d(pos, rot, qi):
        return trac_dist.solve(pos, rot, q_init=qi)

    def trac_l(pos, rot, qi):
        return trac_long.solve(pos, rot, q_init=qi)

    def adam_pos(pos, rot, qi):
        return adam_ik.solve(pos, None, qi, num_seeds=8, max_iters=500, rot_weight=0.0)

    def adam_6dof(pos, rot, qi):
        return adam_ik.solve(pos, rot, qi, num_seeds=8, max_iters=2000, rot_weight=0.1)

    # ===== Test 1: 6-DOF FK-reachable targets (perturbed seed) =====
    print("=" * 70)
    print(f"TEST 1: 6-DOF FK-reachable targets ({NUM_RANDOM} random, perturbed seed)")
    print("=" * 70)
    targets = generate_fk_targets(chain, NUM_RANDOM, ARM)
    r1 = run_test("6DOF-perturbed", targets, [
        ("PinIK-pos", pin_pos),
        ("PinIK-6DOF", pin_6dof),
        ("TracIK-Speed", trac_s),
        ("TracIK-Distance", trac_d),
        ("TracIK-50ms", trac_l),
        ("Adam-pos(8s,500i)", adam_pos),
    ], use_perturbed_seed=True, seed_perturbation=0.1)

    # ===== Test 2: 6-DOF from zero seed (harder) =====
    print("\n" + "=" * 70)
    print(f"TEST 2: 6-DOF FK-reachable targets ({NUM_RANDOM} random, ZERO seed)")
    print("=" * 70)
    r2 = run_test("6DOF-zero", targets, [
        ("PinIK-pos", pin_pos),
        ("PinIK-6DOF", pin_6dof),
        ("TracIK-Speed", trac_s),
        ("TracIK-Distance", trac_d),
        ("TracIK-50ms", trac_l),
        ("Adam-pos(8s,500i)", adam_pos),
    ], use_perturbed_seed=False)

    # ===== Test 3: Large perturbation (far from solution) =====
    print("\n" + "=" * 70)
    print(f"TEST 3: 6-DOF FK-reachable, LARGE perturbation (0.5 rad)")
    print("=" * 70)
    r3 = run_test("6DOF-large-perturb", targets, [
        ("PinIK-pos", pin_pos),
        ("PinIK-6DOF", pin_6dof),
        ("TracIK-Speed", trac_s),
        ("TracIK-Distance", trac_d),
        ("TracIK-50ms", trac_l),
        ("Adam-pos(8s,500i)", adam_pos),
    ], use_perturbed_seed=True, seed_perturbation=0.5)

    # ===== Test 4: Trajectory quality comparison =====
    print("\n" + "=" * 70)
    print("TEST 4: Trajectory quality (8-waypoint path through workspace)")
    print("=" * 70)
    # Pick 8 well-spaced targets that are known reachable
    path_targets = targets[::NUM_RANDOM // 8][:8]
    dt_traj = 0.05

    for sname, sfn in [
        ("PinIK-6DOF", pin_6dof),
        ("TracIK-Speed", trac_s),
        ("TracIK-50ms", trac_l),
        ("Adam-pos", adam_pos),
    ]:
        traj, st, errs, succs = build_trajectory(path_targets, sfn, np.zeros(7), num_interp=20)
        sr = sum(succs) / len(succs)
        total_ms = sum(st) * 1000
        print(f"\n--- {sname} ---")
        print(f"  IK success: {sum(succs)}/{len(succs)} ({sr:.0%})  total={total_ms:.1f}ms")
        if traj is not None and traj.shape[0] > 3:
            analysis = summarize(traj, dt_traj, metadata={"arm": ARM})
            print(f"  Quality: {analysis['quality']}  "
                  f"Path length: {analysis['smoothness']['total_path_length']:.2f} rad  "
                  f"Discontinuities: {len(analysis['discontinuities'])}  "
                  f"Joint limit violations: {len(analysis['joint_limits'])}")
            # Print max velocity per joint
            max_v = [j["max_velocity"] for j in analysis["velocity_acceleration"]["joints"]]
            print(f"  Max vel: {' '.join(f'J{i}={v:.1f}' for i,v in enumerate(max_v))} rad/s")
        else:
            print("  (trajectory build failed — not enough waypoints)")

    # ===== Test 5: Consistency — solve same target 50 times =====
    print("\n" + "=" * 70)
    print("TEST 5: Solution consistency (same target, 50 solves, random seeds)")
    print("=" * 70)
    ref_target = targets[0]
    rng = np.random.default_rng(123)
    nrep = 50

    for sname, sfn in [
        ("PinIK-pos", pin_pos),
        ("TracIK-Speed", trac_s),
        ("Adam-pos", adam_pos),
    ]:
        solutions = []
        errs = []
        for _ in range(nrep):
            qi = rng.uniform(-1, 1, 7).astype(np.float32)
            q, err = sfn(ref_target["pos"], ref_target["rot"], qi)
            if q is not None and err < POS_THRESHOLD:
                solutions.append(np.asarray(q, dtype=np.float32))
                errs.append(err)
        if len(solutions) > 1:
            sols = np.array(solutions)
            std_per_joint = np.std(sols, axis=0)
            print(f"  {sname:<22} success={len(solutions)}/{nrep}  "
                  f"mean_err={np.mean(errs)*1000:.2f}mm  "
                  f"joint_std={np.mean(std_per_joint):.3f}rad  "
                  f"max_joint_std={np.max(std_per_joint):.3f}rad")
        else:
            print(f"  {sname:<22} success={len(solutions)}/{nrep}  (too few to analyze)")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Key findings:")
    print()
    for test_name, results in [("Perturbed seed", r1), ("Zero seed", r2), ("Large perturb", r3)]:
        print(f"  {test_name}:")
        for sname, r in results.items():
            print(f"    {sname:<22} {r['success']:>9}  {r['med_ms']:>6.2f}ms  err={r['mean_err_mm']:.2f}mm")
        print()

    print("Notes:")
    print("  - TracIK requires target rotation (no position-only mode)")
    print("  - PinIK position-only is fastest but can't enforce orientation")
    print("  - Adam is slowest but most robust (multi-seed, GPU-friendly)")
    print("  - For the motion planner pipeline: PinIK-fast → TracIK-6DOF → Adam fallback")


if __name__ == "__main__":
    main()
