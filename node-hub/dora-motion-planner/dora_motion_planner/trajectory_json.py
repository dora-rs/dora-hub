"""Trajectory JSON save/load — no dora dependency.

JSON format (v3 — Linux socketcan wire frames)::

    {
      "version": 3,
      "metadata": {
        "arm": "left",
        "num_joints": 7,
        "num_waypoints": 200,
        "dt": 0.1,
        "motor_kp": [300.0, 300.0, 150.0, 150.0, 40.0, 40.0, 30.0],
        "motor_kd": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "motor_ids": ["0x01", "0x02", ...],
        "created": "2026-02-21T12:00:00+00:00",
        ...
      },
      "commands": [
        {
          "t": 0.0,
          "frames": [
            {"id": "0x01", "data": "<base64 72-byte canfd_frame>"},
            {"id": "0x02", "data": "<base64 72-byte canfd_frame>"},
            ...
          ]
        },
        ...
      ]
    }

Each ``data`` field is ``base64(72 bytes)`` — a Linux ``struct canfd_frame``
ready to be written to the wire::

    struct canfd_frame {        // 72 bytes total, fixed size
        u32 can_id;             // CAN ID + flags (LE)
        u8  len;                // payload length (0-64)
        u8  flags;              // CANFD_BRS=0x01, CANFD_ESI=0x02
        u8  __res0;             // reserved
        u8  __res1;             // reserved
        u8  data[64];           // payload, zero-padded
    };

The ``id`` field is kept for human readability.  Decode with::

    import base64
    wire = base64.b64decode(frame["data"])  # 72 bytes
    can_id = int.from_bytes(wire[0:4], "little") & 0x1FFFFFFF
    data_len = wire[4]
    payload = wire[8:8+data_len]            # 8-byte MIT command

Also reads v2 (8-byte MIT-only ``data``) for backwards compatibility.

Usage::

    from dora_motion_planner.trajectory_json import save, load

    # Save a trajectory (radians → CAN wire frames)
    traj = np.random.randn(200, 7).astype(np.float32)
    save("grasp.json", traj, arm="left", dt=0.1)

    # Load it back (CAN wire frames → radians)
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

# Per-motor gains (motors 1-8), matching openarm_playback.rs
MOTOR_KP = [300.0, 300.0, 150.0, 150.0, 40.0, 40.0, 30.0, 30.0]
MOTOR_KD = [15.0, 15.0, 7.5, 7.5, 2.0, 2.0, 1.5, 1.5]


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


# ---- Linux canfd_frame wire format (72 bytes) ----

CANFD_FRAME_SIZE = 72


def _encode_wire_frame(can_id: int, data: bytes) -> bytes:
    """Encode a CAN frame as a 72-byte Linux ``struct canfd_frame``.

    Layout: [4B can_id LE][1B len][1B flags][2B reserved][64B data zero-padded]
    """
    buf = bytearray(CANFD_FRAME_SIZE)
    buf[0:4] = can_id.to_bytes(4, "little")
    buf[4] = len(data)
    buf[5] = 0  # flags (classic CAN, no BRS/ESI)
    buf[8 : 8 + len(data)] = data
    return bytes(buf)


def _decode_wire_frame(wire: bytes) -> tuple[int, bytes]:
    """Decode a 72-byte ``struct canfd_frame`` into (can_id, payload)."""
    can_id = int.from_bytes(wire[0:4], "little") & 0x1FFFFFFF
    data_len = min(wire[4], 64)
    return can_id, wire[8 : 8 + data_len]


# ---- Save / Load ----

def build(
    trajectory: np.ndarray,
    *,
    arm: str = "left",
    dt: float = 0.1,
    motor_ids: list[int] | None = None,
    gripper: np.ndarray | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Build a (T, J) radian trajectory as a CAN-frame JSON dict (v3).

    Each waypoint is converted to J MIT protocol CAN frames with per-motor
    kp/kd gains from MOTOR_KP/MOTOR_KD, ready to replay on the bus.

    Args:
        trajectory: Joint trajectory in radians, shape ``(T, J)``.
        arm: ``"left"`` or ``"right"``.
        dt: Time step between waypoints in seconds.
        motor_ids: CAN arbitration IDs per joint. Defaults to 0x01..0x07.
        gripper: Optional (T,) array of gripper values, 0.0=open 1.0=closed.
            Encoded as motor 0x08 MIT CAN frame (-1.0472 rad open, 0.0 rad closed).
        extra_metadata: Optional extra fields.

    Returns:
        The JSON-serialisable dict (v3 format).
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
            mid = motor_ids[j]
            kp = MOTOR_KP[mid - 1]
            kd = MOTOR_KD[mid - 1]
            mit = _encode_mit_command(
                p_des=float(traj[i, j]), kp=kp, kd=kd,
            )
            wire = _encode_wire_frame(mid, mit)
            frames.append({
                "id": f"0x{mid:02x}",
                "data": base64.b64encode(wire).decode("ascii"),
            })
        if gripper is not None:
            g = float(gripper[i])
            grip_rad = GRIPPER_OPEN_RAD + g * (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
            kp = MOTOR_KP[GRIPPER_MOTOR_ID - 1]
            kd = MOTOR_KD[GRIPPER_MOTOR_ID - 1]
            mit = _encode_mit_command(p_des=grip_rad, kp=kp, kd=kd)
            wire = _encode_wire_frame(GRIPPER_MOTOR_ID, mit)
            frames.append({
                "id": f"0x{GRIPPER_MOTOR_ID:02x}",
                "data": base64.b64encode(wire).decode("ascii"),
            })
        commands.append({"t": round(i * dt, 6), "frames": frames})

    meta = {
        "arm": arm,
        "num_joints": int(num_joints),
        "num_waypoints": int(num_waypoints),
        "dt": dt,
        "motor_kp": MOTOR_KP[:num_joints],
        "motor_kd": MOTOR_KD[:num_joints],
        "motor_ids": [f"0x{m:02x}" for m in motor_ids],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in meta:
                meta[k] = v

    return {"version": 3, "metadata": meta, "commands": commands}


def save(
    path: str | Path,
    trajectory: np.ndarray,
    *,
    arm: str = "left",
    dt: float = 0.1,
    motor_ids: list[int] | None = None,
    gripper: np.ndarray | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Write a (T, J) radian trajectory as CAN-frame JSON.

    Calls :func:`build` and writes the result to ``path``.

    Returns:
        The JSON-serialisable dict that was written.
    """
    doc = build(
        trajectory,
        arm=arm,
        dt=dt,
        motor_ids=motor_ids,
        gripper=gripper,
        extra_metadata=extra_metadata,
    )

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

    return doc


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a trajectory JSON file.

    Supports v1 (raw float32), v2 (8-byte MIT), and v3 (72-byte wire) formats.

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
        rows = []
        grip_rows = []
        gripper_id = f"0x{GRIPPER_MOTOR_ID:02x}"
        for cmd in commands:
            angles = []
            grip_val = None
            for frame in cmd["frames"]:
                raw = base64.b64decode(frame["data"])
                # v3: 72-byte wire frame → extract MIT payload from bytes 8..16
                # v2: 8-byte MIT payload directly
                if len(raw) == CANFD_FRAME_SIZE:
                    _, mit = _decode_wire_frame(raw)
                else:
                    mit = raw
                pos = _decode_mit_position(mit)
                if frame["id"] == gripper_id:
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
