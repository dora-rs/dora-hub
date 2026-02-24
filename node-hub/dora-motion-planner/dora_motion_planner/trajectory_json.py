"""Trajectory JSON save/load — no dora dependency.

JSON format::

    {
      "version": 2,
      "metadata": {
        "arm": "left",
        "num_joints": 7,
        "num_waypoints": 200,
        "dt": 0.1,
        "kp": 30.0,
        "kd": 1.0,
        "motor_ids": ["0x01", "0x02", ...],
        "created": "2026-02-21T12:00:00+00:00",
        ...
      },
      "commands": [
        {
          "t": 0.0,
          "frames": [
            {"id": "0x01", "data": "<base64 8-byte MIT command>"},
            {"id": "0x02", "data": "<base64 8-byte MIT command>"},
            ...
          ]
        },
        ...
      ]
    }

Each ``data`` field is ``base64(8 bytes)`` — a Damiao MIT protocol CAN frame
ready to be sent on the bus.  Decode with::

    import base64
    raw = base64.b64decode(frame["data"])  # 8 bytes

Usage::

    from dora_motion_planner.trajectory_json import save, load

    # Save a trajectory (radians → CAN frames)
    traj = np.random.randn(200, 7).astype(np.float32)
    save("grasp.json", traj, arm="left", dt=0.1, kp=30.0, kd=1.0)

    # Load it back (CAN frames → radians)
    traj, metadata = load("grasp.json")
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---- Damiao MIT protocol (duplicated from dora_openarm to avoid dependency) ----

Q_MAX = 12.5       # rad
V_MAX = 45.0       # rad/s
T_MAX = 18.0       # Nm
KP_MAX = 500.0
KD_MAX = 5.0

# Default motor IDs for left/right arms (7 joints + gripper on 0x08)
MOTOR_IDS_LEFT = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
MOTOR_IDS_RIGHT = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
GRIPPER_MOTOR_ID = 0x08
GRIPPER_OPEN_RAD = -1.0472   # -60° = fully open (44mm finger travel)
GRIPPER_CLOSED_RAD = 0.0     # fully closed


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    span = x_max - x_min
    x_clamped = max(x_min, min(x_max, x))
    return round((x_clamped - x_min) / span * ((1 << bits) - 1))


def _uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    return x_int / ((1 << bits) - 1) * span + x_min


def _encode_mit_command(
    p_des: float = 0.0,
    v_des: float = 0.0,
    kp: float = 0.0,
    kd: float = 0.0,
    t_ff: float = 0.0,
) -> bytes:
    """Encode a Damiao MIT protocol command (8 bytes)."""
    p = _float_to_uint(p_des, -Q_MAX, Q_MAX, 16)
    v = _float_to_uint(v_des, -V_MAX, V_MAX, 12)
    kp_int = _float_to_uint(kp, 0, KP_MAX, 12)
    kd_int = _float_to_uint(kd, 0, KD_MAX, 12)
    t = _float_to_uint(t_ff, -T_MAX, T_MAX, 12)
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


def _decode_mit_position(data: bytes) -> float:
    """Extract position (radians) from an 8-byte MIT command."""
    p_raw = (data[0] << 8) | data[1]
    return _uint_to_float(p_raw, -Q_MAX, Q_MAX, 16)


# ---- Save / Load ----

def save(
    path: str | Path,
    trajectory: np.ndarray,
    *,
    arm: str = "left",
    dt: float = 0.1,
    kp: float = 30.0,
    kd: float = 1.0,
    motor_ids: list[int] | None = None,
    gripper: np.ndarray | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Write a (T, J) radian trajectory as CAN-frame JSON.

    Each waypoint is converted to J MIT protocol CAN frames with the
    given kp/kd gains, ready to replay on the bus.

    Args:
        path: Output file path.
        trajectory: Joint trajectory in radians, shape ``(T, J)``.
        arm: ``"left"`` or ``"right"``.
        dt: Time step between waypoints in seconds.
        kp: Position gain for the MIT command.
        kd: Damping gain for the MIT command.
        motor_ids: CAN arbitration IDs per joint. Defaults to 0x01..0x07.
        gripper: Optional (T,) array of gripper values, 0.0=open 1.0=closed.
            Encoded as motor 0x08 MIT CAN frame (-1.0472 rad open, 0.0 rad closed).
        extra_metadata: Optional extra fields.

    Returns:
        The JSON-serialisable dict that was written.
    """
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"Expected 2-D trajectory, got shape {traj.shape}")
    num_waypoints, num_joints = traj.shape

    if motor_ids is None:
        motor_ids = list(range(0x01, 0x01 + num_joints))
    if len(motor_ids) != num_joints:
        raise ValueError(
            f"motor_ids length ({len(motor_ids)}) != num_joints ({num_joints})"
        )

    if gripper is not None:
        gripper = np.asarray(gripper, dtype=np.float32)
        if gripper.shape[0] != num_waypoints:
            raise ValueError(
                f"gripper length ({gripper.shape[0]}) != num_waypoints ({num_waypoints})"
            )

    commands = []
    for i in range(num_waypoints):
        frames = []
        for j in range(num_joints):
            raw = _encode_mit_command(
                p_des=float(traj[i, j]), kp=kp, kd=kd,
            )
            frames.append({
                "id": f"0x{motor_ids[j]:02x}",
                "data": base64.b64encode(raw).decode("ascii"),
            })
        if gripper is not None:
            # Map 0.0 (open) → GRIPPER_OPEN_RAD, 1.0 (closed) → GRIPPER_CLOSED_RAD
            g = float(gripper[i])
            grip_rad = GRIPPER_OPEN_RAD + g * (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
            raw = _encode_mit_command(p_des=grip_rad, kp=kp, kd=kd)
            frames.append({
                "id": f"0x{GRIPPER_MOTOR_ID:02x}",
                "data": base64.b64encode(raw).decode("ascii"),
            })
        commands.append({"t": round(i * dt, 6), "frames": frames})

    meta = {
        "arm": arm,
        "num_joints": int(num_joints),
        "num_waypoints": int(num_waypoints),
        "dt": dt,
        "kp": kp,
        "kd": kd,
        "motor_ids": [f"0x{m:02x}" for m in motor_ids],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in meta:
                meta[k] = v

    doc = {"version": 2, "metadata": meta, "commands": commands}

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

    return doc


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a trajectory JSON file.

    Supports both v1 (raw float32) and v2 (CAN frame) formats.

    Returns:
        ``(trajectory, metadata)`` where trajectory is ``(T, J)`` float32 radians.
    """
    with open(path) as f:
        doc = json.load(f)

    meta = doc["metadata"]
    num_joints = int(meta["num_joints"])
    version = doc.get("version", 1)
    commands = doc["commands"]

    if version >= 2:
        # Decode MIT CAN frames back to radians
        rows = []
        grip_rows = []
        gripper_id = f"0x{GRIPPER_MOTOR_ID:02x}"
        for cmd in commands:
            angles = []
            grip_val = None
            for frame in cmd["frames"]:
                raw = base64.b64decode(frame["data"])
                pos = _decode_mit_position(raw)
                if frame["id"] == gripper_id:
                    # Convert motor radians back to 0-1 scale
                    grip_val = (pos - GRIPPER_OPEN_RAD) / (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
                else:
                    angles.append(pos)
            rows.append(angles)
            grip_rows.append(grip_val)
        traj = np.array(rows, dtype=np.float32)
        has_gripper = any(g is not None for g in grip_rows)
        if has_gripper:
            meta["gripper"] = np.array(
                [g if g is not None else 0.0 for g in grip_rows], dtype=np.float32
            )
    else:
        # v1: raw base64 float32
        traj = np.stack([
            np.frombuffer(base64.b64decode(cmd["data"]), dtype=np.float32)
            for cmd in commands
        ])

    if traj.shape[1] != num_joints:
        raise ValueError(
            f"Joint count mismatch: data has {traj.shape[1]}, "
            f"metadata says {num_joints}"
        )
    return traj, meta
