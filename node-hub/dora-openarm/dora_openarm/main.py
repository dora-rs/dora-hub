"""Dora CAN bus driver for OpenArm — Damiao motors via MIT protocol.

Reads motor states on tick, publishes joint angles. Accepts joint commands
from an IK node. Starts in read-only mode (zero-torque queries only).

A velocity/acceleration safety limiter catches spikes from any trajectory
source.  If any joint exceeds MAX_JOINT_VEL the entire step is uniformly
scaled (preserving Cartesian direction).  Acceleration is dampened similarly.

Env vars:
    CAN_INTERFACE: socketcan interface name (default "can2")
    MOTOR_IDS: comma-separated hex CAN IDs (default "0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18")
    KP: position gain (default 0 = read-only)
    KD: damping gain (default 0 = read-only)
    RECV_TIMEOUT: CAN recv timeout in seconds (default 0.01)
    MAX_JOINT_VEL: per-joint velocity limit in rad/s (default 3.0)
    MAX_JOINT_ACC: per-joint acceleration limit in rad/s² (default 15.0)
"""

import os
import time

import numpy as np
import pyarrow as pa

# Damiao MIT protocol constants
Q_MAX = 12.5       # rad
V_MAX = 45.0       # rad/s
T_MAX = 18.0       # Nm
KP_MAX = 500.0
KD_MAX = 5.0

ENABLE_CMD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
DISABLE_CMD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])

# Joint limits from URDF (L_J1 through L_J7 — same structure for right arm with mirrored limits)
# Format: (lower, upper) in radians
JOINT_LIMITS_LEFT = [
    (-3.490659, 1.396263),   # L_J1
    (-3.316125, 0.174533),   # L_J2
    (-1.570796, 1.570796),   # L_J3
    (0.0, 2.443461),         # L_J4
    (-1.570796, 1.570796),   # L_J5
    (-0.785398, 0.785398),   # L_J6
    (-1.570796, 1.570796),   # L_J7
]

JOINT_LIMITS_RIGHT = [
    (-1.396263, 3.490659),   # R_J1
    (-0.174533, 3.316125),   # R_J2
    (-1.570796, 1.570796),   # R_J3
    (0.0, 2.443461),         # R_J4
    (-1.570796, 1.570796),   # R_J5
    (-0.785398, 0.785398),   # R_J6
    (-1.570796, 1.570796),   # R_J7
]

# Gripper conversion: motor radians -> URDF prismatic meters
GRIPPER_MOTOR_OPEN_RAD = -1.0472  # -60 deg = -pi/3
GRIPPER_JOINT_OPEN_M = 0.044      # 44mm finger travel


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Map a float from [x_min, x_max] to unsigned int [0, 2^bits - 1]."""
    span = x_max - x_min
    x_clamped = max(x_min, min(x_max, x))
    return round((x_clamped - x_min) / span * ((1 << bits) - 1))


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """Map an unsigned int [0, 2^bits - 1] to float [x_min, x_max]."""
    span = x_max - x_min
    return x_int / ((1 << bits) - 1) * span + x_min


def parse_damiao_state(data: bytes):
    """Parse Damiao motor 8-byte state response.

    Returns (position_rad, velocity_rad_s, torque_nm, temp_mos, temp_rotor)
    or None if data is too short.
    """
    if len(data) < 8:
        return None
    # data[0] is motor_id (unused — already known from CAN arbitration ID)
    q_raw = (data[1] << 8) | data[2]
    dq_raw = (data[3] << 4) | (data[4] >> 4)
    tau_raw = ((data[4] & 0x0F) << 8) | data[5]

    q_rad = uint_to_float(q_raw, -Q_MAX, Q_MAX, 16)
    vel = uint_to_float(dq_raw, -V_MAX, V_MAX, 12)
    tau = uint_to_float(tau_raw, -T_MAX, T_MAX, 12)

    return q_rad, vel, tau, data[6], data[7]


def encode_mit_command(
    p_des: float = 0.0,
    v_des: float = 0.0,
    kp: float = 0.0,
    kd: float = 0.0,
    t_ff: float = 0.0,
) -> bytes:
    """Encode a Damiao MIT protocol command (8 bytes).

    With kp=0, kd=0, t_ff=0 this is a zero-torque query that reads state
    without applying any force.
    """
    p = float_to_uint(p_des, -Q_MAX, Q_MAX, 16)
    v = float_to_uint(v_des, -V_MAX, V_MAX, 12)
    kp_int = float_to_uint(kp, 0, KP_MAX, 12)
    kd_int = float_to_uint(kd, 0, KD_MAX, 12)
    t = float_to_uint(t_ff, -T_MAX, T_MAX, 12)

    return bytes([
        (p >> 8) & 0xFF,
        p & 0xFF,
        (v >> 4) & 0xFF,
        ((v & 0xF) << 4) | ((kp_int >> 8) & 0xF),
        kp_int & 0xFF,
        (kd_int >> 4) & 0xFF,
        ((kd_int & 0xF) << 4) | ((t >> 8) & 0xF),
        t & 0xFF,
    ])


# Pre-computed zero-torque query command
ZERO_TORQUE_CMD = encode_mit_command()


def check_joint_limits(angles: np.ndarray, limits: list) -> bool:
    """Check if all joint angles are within URDF limits."""
    for i, (lo, hi) in enumerate(limits):
        if i >= len(angles):
            break
        if angles[i] < lo or angles[i] > hi:
            print(f"Joint {i+1} angle {angles[i]:.4f} out of limits [{lo:.4f}, {hi:.4f}]")
            return False
    return True


def main():
    """Dora OpenArm CAN driver node."""
    import can
    from dora import Node

    # Configuration from env
    mock_mode = os.getenv("MOCK", "false").lower() in ("1", "true", "yes")
    can_interface = os.getenv("CAN_INTERFACE", "can2")
    motor_ids_str = os.getenv("MOTOR_IDS", "0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08")
    motor_ids = [int(x.strip(), 16) for x in motor_ids_str.split(",")]
    # Response ID mapping: accept both cmd_id and cmd_id + 0x10
    # (real motors respond on cmd+0x10, fake-can-server responds on same ID)
    resp_id_to_idx = {}
    for i, mid in enumerate(motor_ids):
        resp_id_to_idx[mid] = i
        resp_id_to_idx[mid + 0x10] = i
    kp = float(os.getenv("KP", "0"))
    kd = float(os.getenv("KD", "0"))
    recv_timeout = float(os.getenv("RECV_TIMEOUT", "0.5"))
    arm_side = os.getenv("ARM_SIDE", "left")  # "left" or "right"
    joint_limits = JOINT_LIMITS_LEFT if arm_side == "left" else JOINT_LIMITS_RIGHT

    num_joints = 7  # 7 DOF arm joints (gripper is motor_ids[7])
    enabled = False
    read_only = kp == 0 and kd == 0
    auto_enable = os.getenv("AUTO_ENABLE", "false").lower() in ("1", "true", "yes")

    print(f"OpenArm CAN driver: interface={can_interface}, motors={[hex(m) for m in motor_ids]}")
    print(f"  arm_side={arm_side}, kp={kp}, kd={kd}, read_only={read_only}, mock={mock_mode}")

    # Open CAN bus (skip in mock mode)
    bus = None
    if not mock_mode:
        bus = can.interface.Bus(channel=can_interface, interface="socketcan", fd=False)

    node = Node()

    # Current state arrays
    joint_positions = np.zeros(num_joints + 1, dtype=np.float32)  # 7 joints + gripper
    joint_velocities = np.zeros(num_joints + 1, dtype=np.float32)
    joint_torques = np.zeros(num_joints + 1, dtype=np.float32)
    joint_temps = np.zeros((num_joints + 1, 2), dtype=np.float32)  # [mos, rotor] per joint

    # Velocity / acceleration limiter state
    max_joint_vel = float(os.getenv("MAX_JOINT_VEL", "3.0"))
    max_joint_acc = float(os.getenv("MAX_JOINT_ACC", "15.0"))
    _last_cmd = [None]       # last commanded position (np array or None)
    _last_cmd_vel = [None]   # last commanded velocity  (np array or None)
    _last_cmd_time = [0.0]

    if auto_enable and not read_only and bus is not None:
        print("Auto-enabling motors...")
        for mid in motor_ids:
            msg = can.Message(arbitration_id=mid, data=ENABLE_CMD, is_fd=False)
            bus.send(msg)
            bus.recv(timeout=recv_timeout)
        enabled = True
        _last_cmd[0] = None  # reset limiter state
        print("Motors enabled")

    def limit_command(target: np.ndarray) -> np.ndarray:
        """Clamp velocity and acceleration to safe limits.

        Uniform scaling preserves Cartesian direction when any joint is
        over the velocity limit.  Acceleration is dampened per-step.
        """
        now = time.perf_counter()
        if _last_cmd[0] is None:
            _last_cmd[0] = target.copy()
            _last_cmd_vel[0] = np.zeros_like(target)
            _last_cmd_time[0] = now
            return target

        dt = now - _last_cmd_time[0]
        if dt < 1e-6:
            return _last_cmd[0]  # dt too small, hold previous

        delta = target - _last_cmd[0]
        vel = delta / dt

        # --- velocity limit (uniform scale) ---
        vel_abs = np.abs(vel)
        max_v = np.max(vel_abs)
        if max_v > max_joint_vel:
            scale = max_joint_vel / max_v
            vel *= scale
            print(f"[LIMITER] vel clamp: scale={scale:.3f} (peak={max_v:.2f} rad/s)")

        # --- acceleration limit (dampen velocity change) ---
        acc = (vel - _last_cmd_vel[0]) / dt
        acc_abs = np.abs(acc)
        max_a = np.max(acc_abs)
        if max_a > max_joint_acc:
            scale_a = max_joint_acc / max_a
            vel = _last_cmd_vel[0] + (vel - _last_cmd_vel[0]) * scale_a
            print(f"[LIMITER] acc clamp: scale={scale_a:.3f} (peak={max_a:.2f} rad/s²)")

        limited = _last_cmd[0] + vel * dt
        _last_cmd[0] = limited.copy()
        _last_cmd_vel[0] = vel.copy()
        _last_cmd_time[0] = now
        return limited

    # Timing stats
    _tick_times = []
    _query_times = []
    _cmd_times = []
    _last_tick = [0.0]
    _tick_count = [0]

    def query_all_motors():
        """Send zero-torque query to all motors and collect responses."""
        t0 = time.perf_counter()
        n_resp = 0
        for mid in motor_ids:
            msg = can.Message(arbitration_id=mid, data=ZERO_TORQUE_CMD, is_fd=False)
            try:
                bus.send(msg)
            except can.CanError as e:
                print(f"CAN send error for motor {hex(mid)}: {e}")
                continue

        # Collect responses
        for _ in range(len(motor_ids)):
            resp = bus.recv(timeout=recv_timeout)
            if resp is None:
                continue
            n_resp += 1
            state = parse_damiao_state(bytes(resp.data))
            if state is None:
                continue
            resp_id = resp.arbitration_id
            if resp_id in resp_id_to_idx:
                idx = resp_id_to_idx[resp_id]
                joint_positions[idx] = state[0]
                joint_velocities[idx] = state[1]
                joint_torques[idx] = state[2]
                joint_temps[idx] = [state[3], state[4]]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _query_times.append(elapsed_ms)

        # Log every 10 ticks
        _tick_count[0] += 1
        if _tick_count[0] % 10 == 0:
            avg_q = np.mean(_query_times[-10:])
            avg_c = np.mean(_cmd_times[-10:]) if _cmd_times[-10:] else 0
            intervals = _tick_times[-10:]
            avg_dt = np.mean(intervals) if intervals else 0
            jitter = np.std(intervals) if len(intervals) > 1 else 0
            print(f"[KPI] tick_dt={avg_dt:.1f}ms jitter={jitter:.1f}ms "
                  f"query={avg_q:.1f}ms cmd={avg_c:.1f}ms resp={n_resp}/{len(motor_ids)}")

    def send_joint_command(target_angles: np.ndarray):
        """Send MIT position commands for 7 arm joints."""
        if not enabled:
            print("Motors not enabled — ignoring joint command")
            return
        if read_only:
            print("Read-only mode (KP=0, KD=0) — ignoring joint command")
            return

        # Check joint limits for arm joints only (first 7)
        arm_angles = target_angles[:num_joints]
        if not check_joint_limits(arm_angles, joint_limits):
            print("Joint command rejected: exceeds limits")
            return

        # Safety limiter: clamp velocity and acceleration
        target_angles = target_angles.copy()
        target_angles[:num_joints] = limit_command(target_angles[:num_joints])

        for i in range(min(num_joints, len(target_angles))):
            cmd = encode_mit_command(
                p_des=target_angles[i],
                v_des=0.0,
                kp=kp,
                kd=kd,
                t_ff=0.0,
            )
            msg = can.Message(
                arbitration_id=motor_ids[i],
                data=cmd,
                is_fd=False,
            )
            try:
                bus.send(msg)
            except can.CanError as e:
                print(f"CAN send error for motor {hex(motor_ids[i])}: {e}")

        # Drain responses from position commands
        for _ in range(min(num_joints, len(target_angles))):
            bus.recv(timeout=recv_timeout)

        print(f"Sent joint cmd: [{', '.join(f'{a:.3f}' for a in target_angles[:num_joints])}]")

    try:
        for event in node:
            if event["type"] == "INPUT":
                event_id = event["id"]

                if event_id == "tick":
                    now = time.perf_counter()
                    if _last_tick[0] > 0:
                        _tick_times.append((now - _last_tick[0]) * 1000)
                    _last_tick[0] = now
                    if not mock_mode:
                        query_all_motors()
                    # Publish joint_state: float32 array of positions (radians)
                    metadata = event["metadata"]
                    metadata["encoding"] = "jointstate"
                    node.send_output(
                        "joint_state",
                        pa.array(joint_positions, type=pa.float32()),
                        metadata=metadata,
                    )

                elif event_id in ("joint_command", "home_command"):
                    # Filter by arm side if metadata specifies one
                    cmd_arm = metadata.get("arm", "")
                    if cmd_arm and cmd_arm != arm_side:
                        continue
                    t0 = time.perf_counter()
                    target = event["value"].to_numpy().astype(np.float32)
                    send_joint_command(target)
                    _cmd_times.append((time.perf_counter() - t0) * 1000)

                elif event_id == "gripper_command":
                    cmd_arm = metadata.get("arm", "")
                    if cmd_arm and cmd_arm != arm_side:
                        continue
                    target_rad = event["value"].to_numpy().astype(np.float32)[0]
                    target_rad = max(GRIPPER_MOTOR_OPEN_RAD, min(0.0, target_rad))
                    if not enabled:
                        print("Motors not enabled — ignoring gripper command")
                    elif read_only:
                        print("Read-only mode — ignoring gripper command")
                    else:
                        cmd = encode_mit_command(p_des=target_rad, kp=kp, kd=kd)
                        msg = can.Message(
                            arbitration_id=motor_ids[7],
                            data=cmd,
                            is_fd=False,
                        )
                        try:
                            bus.send(msg)
                            bus.recv(timeout=recv_timeout)
                            print(f"Gripper command: {target_rad:.4f} rad")
                        except can.CanError as e:
                            print(f"CAN send error for gripper motor: {e}")

                elif event_id == "trigger":
                    cmd_text = event["value"][0].as_py().strip().lower()
                    if cmd_text == "enable":
                        print("Enabling motors...")
                        for mid in motor_ids:
                            msg = can.Message(arbitration_id=mid, data=ENABLE_CMD, is_fd=False)
                            bus.send(msg)
                            bus.recv(timeout=recv_timeout)
                        enabled = True
                        _last_cmd[0] = None  # reset limiter state
                        _last_cmd_vel[0] = None
                        print("Motors enabled")
                    elif cmd_text == "disable":
                        print("Disabling motors...")
                        for mid in motor_ids:
                            msg = can.Message(arbitration_id=mid, data=DISABLE_CMD, is_fd=False)
                            bus.send(msg)
                            bus.recv(timeout=recv_timeout)
                        enabled = False
                        print("Motors disabled")
                    elif cmd_text == "query":
                        query_all_motors()
                        print(f"Positions (rad): {joint_positions}")
                        print(f"Velocities (rad/s): {joint_velocities}")
                        print(f"Torques (Nm): {joint_torques}")

            elif event["type"] == "STOP":
                print("Received STOP — disabling motors")
                if bus is not None:
                    for mid in motor_ids:
                        msg = can.Message(arbitration_id=mid, data=DISABLE_CMD, is_fd=False)
                        try:
                            bus.send(msg)
                        except can.CanError:
                            pass
                break
    finally:
        # Always disable motors on exit
        if bus is not None:
            for mid in motor_ids:
                try:
                    msg = can.Message(arbitration_id=mid, data=DISABLE_CMD, is_fd=False)
                    bus.send(msg)
                except Exception:
                    pass
            bus.shutdown()
            print("CAN bus shutdown complete")
        else:
            print("Mock mode shutdown complete")


if __name__ == "__main__":
    main()
