"""Compiled forward kinematics for OpenArm using pure tensor ops.

Replaces pytorch_kinematics FK with a torch.compile-friendly function
that fuses the entire FK chain into fewer kernel dispatches.  This is
critical for Apple MPS where each kernel launch costs ~0.5-1ms overhead.

The chain has 13 frames: 3 fixed prefix + 7 revolute + 3 fixed suffix.
Consecutive fixed frames are pre-multiplied, so at runtime the FK is:

    T_ee = T_prefix @ R1(q1) @ O1 @ R2(q2) @ O2 @ ... @ R7(q7) @ T_suffix

where T_prefix, O_i, T_suffix are constant (4,4) matrices and R_i(q)
is a rotation about a known axis.

The `end_only=False` variant also returns all intermediate link transforms,
needed for capsule collision checking.
"""

import torch

# ---------- Extracted from OpenArm URDF via extract_fk_params.py ----------
# Frame 0 (world): identity
# Frame 1 (body_link0): rotation about Z by ~-0.7854 rad  (fixed)
# Frame 2 (left_link0): position=(0, 0.031, 0.698), rot=Rx(-90°)  (fixed)
# Frames 3-9: revolute joints with offsets
# Frames 10-12: link8, hand, tcp  (all fixed, translations only)

# Raw offset matrices at q=0 for each frame (extracted via extract_fk_params.py).
_RAW_OFFSETS = [
    # Frame 0: world (identity)
    [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
    # Frame 1: body_link0 (fixed rotation)
    [[-0.70979238, 0.70441079, 0., 0.], [-0.70441079, -0.70979238, 0., 0.],
     [0., 0., 1., 0.], [0., 0., 0., 1.]],
    # Frame 2: left_link0 (fixed, pos + Rx(-90°))
    [[1., 0., 0., 0.], [0., -3.6955e-06, 1., 0.031],
     [0., -1., -3.6955e-06, 0.698], [0., 0., 0., 1.]],
    # Frame 3: L_J1 revolute (offset = translation z=0.0625)
    [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.0625], [0., 0., 0., 1.]],
    # Frame 4: L_J2 revolute (offset includes Ry(-90°) style transform)
    [[1., 0., 0., -0.0301], [0., 0., 1., 0.], [0., -1., 0., 0.06], [0., 0., 0., 1.]],
    # Frame 5: L_J3 revolute
    [[1., 0., 0., 0.0301], [0., 1., 0., 0.], [0., 0., 1., 0.11625], [0., 0., 0., 1.]],
    # Frame 6: L_J4 revolute
    [[1., 0., 0., 0.], [0., 1., 0., 0.0315], [0., 0., 1., 0.15375], [0., 0., 0., 1.]],
    # Frame 7: L_J5 revolute
    [[1., 0., 0., 0.], [0., 1., 0., -0.0315], [0., 0., 1., 0.0955], [0., 0., 0., 1.]],
    # Frame 8: L_J6 revolute
    [[1., 0., 0., 0.0375], [0., 1., 0., 0.], [0., 0., 1., 0.1205], [0., 0., 0., 1.]],
    # Frame 9: L_J7 revolute
    [[1., 0., 0., -0.0375], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
    # Frame 10: link8 (fixed, tiny translation)
    [[1., 0., 0., 1e-06], [0., 1., 0., 0.0205], [0., 0., 1., 0.], [0., 0., 0., 1.]],
    # Frame 11: hand (fixed, translation)
    [[1., 0., 0., 0.], [0., 1., 0., -0.025], [0., 0., 1., 0.1001], [0., 0., 0., 1.]],
    # Frame 12: hand_tcp (fixed, translation)
    [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.08], [0., 0., 0., 1.]],
]

_IS_REVOLUTE = [False, False, False, True, True, True, True, True, True, True, False, False, False]
_AXES = [
    [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],  # fixed prefix
    [0., 0., 1.], [-1., 0., 0.], [0., 0., 1.],  # J1(Z), J2(-X), J3(Z)
    [0., 1., 0.], [0., 0., 1.], [1., 0., 0.],   # J4(Y), J5(Z), J6(X)
    [0., -1., 0.],                                # J7(-Y)
    [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],   # fixed suffix
]

# Link names corresponding to each frame (for collision model compatibility).
LINK_NAMES = [
    "world", "openarm_body_link0", "openarm_left_link0",
    "openarm_left_link1", "openarm_left_link2", "openarm_left_link3",
    "openarm_left_link4", "openarm_left_link5", "openarm_left_link6",
    "openarm_left_link7", "openarm_left_link8", "openarm_left_hand",
    "openarm_left_hand_tcp",
]


def _axis_angle_to_rot_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: axis (3,), angle (B,) -> (B, 3, 3).

    Assumes axis is a unit vector.
    """
    B = angle.shape[0]
    c = torch.cos(angle)  # (B,)
    s = torch.sin(angle)  # (B,)
    t = 1.0 - c

    x, y, z = axis[0], axis[1], axis[2]

    # Build rotation matrices directly
    R = torch.zeros(B, 3, 3, device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = t * x * x + c
    R[:, 0, 1] = t * x * y - s * z
    R[:, 0, 2] = t * x * z + s * y
    R[:, 1, 0] = t * x * y + s * z
    R[:, 1, 1] = t * y * y + c
    R[:, 1, 2] = t * y * z - s * x
    R[:, 2, 0] = t * x * z - s * y
    R[:, 2, 1] = t * y * z + s * x
    R[:, 2, 2] = t * z * z + c
    return R


def _rot_to_4x4(R: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, 3) rotation to (B, 4, 4) homogeneous transform."""
    B = R.shape[0]
    T = torch.zeros(B, 4, 4, device=R.device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, 3, 3] = 1.0
    return T


class CompiledFK:
    """Compiled forward kinematics for OpenArm left arm.

    Pre-multiplies consecutive fixed frames into single transforms
    and stores joint axes as tensors.  The FK computation uses only
    basic torch ops (matmul, cos, sin, indexing) that torch.compile
    can trace and fuse.

    Usage::

        fk = CompiledFK(device="mps")
        # End-effector only (fast):
        ee_mat = fk.forward(q)  # q: (B, 7), returns (B, 4, 4)
        # All links (for collision):
        link_mats = fk.forward_all(q)  # returns dict[str, (B, 4, 4)]
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)

        offsets = [torch.tensor(o, dtype=torch.float32, device=self.device)
                   for o in _RAW_OFFSETS]
        axes = [torch.tensor(a, dtype=torch.float32, device=self.device)
                for a in _AXES]

        # Pre-multiply the 3 fixed prefix frames into one transform.
        prefix = offsets[0] @ offsets[1] @ offsets[2]
        self.prefix: torch.Tensor = prefix  # (4, 4)

        # For each revolute joint, store its offset transform and axis.
        # Revolute frames are indices 3..9 (7 joints).
        self.joint_offsets: list[torch.Tensor] = []  # each (4, 4)
        self.joint_axes: list[torch.Tensor] = []      # each (3,)
        for i in range(3, 10):
            self.joint_offsets.append(offsets[i])
            self.joint_axes.append(axes[i])

        # Pre-multiply the 3 fixed suffix frames (10, 11, 12).
        suffix = offsets[10] @ offsets[11] @ offsets[12]
        self.suffix: torch.Tensor = suffix  # (4, 4)

        # For forward_all: we also need individual suffix frames
        # and cumulative prefix at each link.
        self._offsets = offsets
        self._link_names = LINK_NAMES

        # Pre-compute which link indices correspond to revolute frames
        # Link transforms needed for collision: "openarm_left_link0" through TCP
        # Frame i -> link LINK_NAMES[i]
        self._revolute_indices = [3, 4, 5, 6, 7, 8, 9]
        self._fixed_suffix_indices = [10, 11, 12]

        # Stack joint offsets and axes for batched computation
        self._stacked_offsets = torch.stack(self.joint_offsets)  # (7, 4, 4)
        self._stacked_axes = torch.stack(self.joint_axes)        # (7, 3)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute end-effector transform only.

        Args:
            q: (B, 7) joint angles.

        Returns:
            (B, 4, 4) end-effector homogeneous transform.
        """
        B = q.shape[0]
        # Start with prefix (broadcast to batch)
        T = self.prefix.unsqueeze(0).expand(B, 4, 4).clone()

        for j in range(7):
            # Apply offset then rotation
            T = T @ self.joint_offsets[j]
            R = _axis_angle_to_rot_matrix(self.joint_axes[j], q[:, j])
            R4 = _rot_to_4x4(R)
            T = T @ R4

        # Apply suffix
        T = T @ self.suffix
        return T

    def forward_all_stacked(self, q: torch.Tensor) -> torch.Tensor:
        """Compute transforms for all 13 links as a single stacked tensor.

        This is the compile-friendly variant — no dicts, no clones, just
        writes into a pre-allocated (13, B, 4, 4) output tensor.

        Args:
            q: (B, 7) joint angles.

        Returns:
            (13, B, 4, 4) tensor of homogeneous transforms for each link.
        """
        B = q.shape[0]
        out = torch.zeros(13, B, 4, 4, device=q.device, dtype=q.dtype)

        # Fixed prefix: frames 0, 1, 2
        T = self._offsets[0].unsqueeze(0).expand(B, 4, 4)
        out[0] = T
        T = T @ self._offsets[1]
        out[1] = T
        T = T @ self._offsets[2]
        out[2] = T

        # Revolute joints: frames 3-9
        for j in range(7):
            T = T @ self.joint_offsets[j]
            R = _axis_angle_to_rot_matrix(self.joint_axes[j], q[:, j])
            R4 = _rot_to_4x4(R)
            T = T @ R4
            out[3 + j] = T

        # Fixed suffix: frames 10, 11, 12
        for k, idx in enumerate(self._fixed_suffix_indices):
            T = T @ self._offsets[idx]
            out[idx] = T

        return out

    def forward_all(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute transforms for all links (for collision checking).

        Internally calls :meth:`forward_all_stacked` and unpacks into a
        dict keyed by link name.
        """
        stacked = self.forward_all_stacked(q)
        return {
            self._link_names[i]: stacked[i]
            for i in range(13)
        }


class _Transform3dShim:
    """Minimal shim that wraps a (B, 4, 4) tensor with a get_matrix() method.

    This makes CompiledFK output compatible with code that expects
    pytorch_kinematics Transform3d objects (which provide get_matrix()).
    """

    __slots__ = ("_mat",)

    def __init__(self, mat: torch.Tensor):
        self._mat = mat

    def get_matrix(self) -> torch.Tensor:
        return self._mat


class CompiledFKAdapter:
    """Drop-in replacement for ``pk.Chain`` in TrajectoryOptimizer.

    Wraps :class:`CompiledFK` to provide the same ``forward_kinematics``
    API that returns ``dict[str, Transform3d]``-like objects.
    """

    def __init__(self, compiled_fk: CompiledFK):
        self._fk = compiled_fk
        self._link_names = LINK_NAMES
        self.device = compiled_fk.device

    def forward_kinematics(
        self, q: torch.Tensor, end_only: bool = True
    ) -> dict[str, _Transform3dShim] | _Transform3dShim:
        if end_only:
            mat = self._fk.forward(q)
            return _Transform3dShim(mat)
        else:
            mats = self._fk.forward_all(q)
            return {name: _Transform3dShim(m) for name, m in mats.items()}

    def get_link_names(self) -> list[str]:
        return list(self._link_names)

    def to(self, *args, **kwargs):
        """No-op for API compatibility (device already set at init)."""
        return self
