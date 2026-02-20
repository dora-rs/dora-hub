"""Gradient-based trajectory optimisation with differentiable FK and collision costs.

Multi-start Adam optimisation over a trajectory tensor `q_traj` of shape (T, 7).
Costs: smoothness (acceleration), environment collision (capsule–point cloud),
self-collision (capsule–capsule), and joint limits.
"""

import torch
import pytorch_kinematics as pk

from .collision_model import (
    CapsuleCollisionModel,
    SELF_COLLISION_PAIRS,
    capsule_points_distance,
    capsule_capsule_distance,
)


class TrajectoryOptimizer:
    """Gradient-based trajectory planner with collision avoidance."""

    def __init__(
        self,
        chain: pk.Chain,
        capsule_model: CapsuleCollisionModel,
        joint_limits: tuple[list[float], list[float]],
        device: str | torch.device = "cuda",
        safety_margin: float = 0.02,
    ):
        self.device = torch.device(device)
        # pk's chain.to() has a bug with device/dtype kwargs on some versions;
        # move via dtype+device explicitly to avoid it.
        self.chain = chain.to(dtype=torch.float32, device=str(self.device))
        self.capsules = capsule_model
        self.lower = torch.tensor(
            joint_limits[0], dtype=torch.float32, device=self.device
        )
        self.upper = torch.tensor(
            joint_limits[1], dtype=torch.float32, device=self.device
        )
        self.safety_margin = safety_margin

        # Cost weights
        self.w_smooth = 10.0
        self.w_env = 100.0
        self.w_self = 50.0
        self.w_limits = 1000.0

    def optimize(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        point_cloud: torch.Tensor | None = None,
        T: int = 200,
        num_seeds: int = 8,
        max_iters: int = 500,
        lr: float = 0.01,
    ) -> tuple[torch.Tensor, float]:
        """Multi-start gradient-based trajectory optimisation.

        Args:
            q_start: (7,) start joint configuration.
            q_goal: (7,) goal joint configuration.
            point_cloud: (N, 3) obstacle points in robot base frame, or None.
            T: Number of waypoints.
            num_seeds: Number of random initialisations.
            max_iters: Adam iterations per seed.
            lr: Learning rate.

        Returns:
            (best_traj, best_cost) — best_traj is (T, 7) numpy array.
        """
        q_start = q_start.to(self.device).detach()
        q_goal = q_goal.to(self.device).detach()
        if point_cloud is not None:
            point_cloud = point_cloud.to(self.device).detach()

        best_cost = float("inf")
        best_traj = None

        for seed in range(num_seeds):
            # Linear interpolation
            t = torch.linspace(0, 1, T, device=self.device).unsqueeze(1)  # (T, 1)
            q_init = q_start + t * (q_goal - q_start)  # (T, 7)

            if seed > 0:
                noise = torch.randn_like(q_init) * 0.3
                q_init = q_init + noise
                q_init = torch.clamp(q_init, self.lower, self.upper)

            # Fix endpoints — only optimise interior waypoints
            q_inner = q_init[1:-1].clone().requires_grad_(True)
            optimizer = torch.optim.Adam([q_inner], lr=lr)

            for it in range(max_iters):
                optimizer.zero_grad()
                full_traj = torch.cat(
                    [q_start.unsqueeze(0), q_inner, q_goal.unsqueeze(0)]
                )
                cost = self._total_cost(full_traj, point_cloud)
                cost.backward()
                optimizer.step()

                with torch.no_grad():
                    q_inner.clamp_(self.lower, self.upper)

            with torch.no_grad():
                final_traj = torch.cat(
                    [q_start.unsqueeze(0), q_inner, q_goal.unsqueeze(0)]
                )
                final_cost = self._total_cost(final_traj, point_cloud).item()
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_traj = final_traj.cpu()

        # Hard-clamp entire trajectory to joint limits (endpoints may come from IK)
        if best_traj is not None:
            best_traj = torch.clamp(best_traj, self.lower.cpu(), self.upper.cpu())

        return best_traj, best_cost

    def _total_cost(
        self, q_traj: torch.Tensor, point_cloud: torch.Tensor | None
    ) -> torch.Tensor:
        """Compute total differentiable cost over trajectory."""
        # 1. Smoothness: minimise acceleration
        vel = q_traj[1:] - q_traj[:-1]
        acc = vel[1:] - vel[:-1]
        smooth_cost = (acc**2).sum()

        # 2. FK for all waypoints (batched)
        transforms = self.chain.forward_kinematics(q_traj, end_only=False)

        # 3. Environment collision cost
        env_cost = torch.tensor(0.0, device=self.device)
        if point_cloud is not None and len(point_cloud) > 0:
            env_cost = self._env_collision_cost(transforms, point_cloud)

        # 4. Self-collision cost
        self_cost = self._self_collision_cost(transforms)

        # 5. Joint limit penalty (soft barrier)
        limit_cost = (torch.relu(self.lower - q_traj) ** 2).sum() + (
            torch.relu(q_traj - self.upper) ** 2
        ).sum()

        return (
            self.w_smooth * smooth_cost
            + self.w_env * env_cost
            + self.w_self * self_cost
            + self.w_limits * limit_cost
        )

    def _env_collision_cost(
        self, transforms: dict, point_cloud: torch.Tensor
    ) -> torch.Tensor:
        """Hinge-loss collision cost between all link capsules and point cloud."""
        margin = self.safety_margin
        cost = torch.tensor(0.0, device=self.device)

        for link_name in self.capsules.link_names:
            if link_name not in transforms:
                continue
            p0_w, p1_w = self.capsules.capsule_endpoints_world(link_name, transforms)
            radius = self.capsules.radii[link_name]
            # signed distance: positive = free, negative = penetrating
            sd = capsule_points_distance(p0_w, p1_w, radius, point_cloud)  # (B, N)
            # Hinge loss with safety margin
            violation = torch.relu(margin - sd)  # (B, N)
            cost = cost + (violation**2).sum()

        return cost

    def _self_collision_cost(self, transforms: dict) -> torch.Tensor:
        """Hinge-loss self-collision cost between non-adjacent link pairs."""
        margin = self.safety_margin
        cost = torch.tensor(0.0, device=self.device)
        link_names = self.capsules.link_names

        for i, j in SELF_COLLISION_PAIRS:
            name_i = link_names[i]
            name_j = link_names[j]
            if name_i not in transforms or name_j not in transforms:
                continue

            p0_i, p1_i = self.capsules.capsule_endpoints_world(name_i, transforms)
            r_i = self.capsules.radii[name_i]
            p0_j, p1_j = self.capsules.capsule_endpoints_world(name_j, transforms)
            r_j = self.capsules.radii[name_j]

            sd = capsule_capsule_distance(p0_i, p1_i, r_i, p0_j, p1_j, r_j)  # (B,)
            violation = torch.relu(margin - sd)
            cost = cost + (violation**2).sum()

        return cost
