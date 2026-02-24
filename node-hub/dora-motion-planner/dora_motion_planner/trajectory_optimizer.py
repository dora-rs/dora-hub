"""Gradient-based trajectory optimisation with differentiable FK and collision costs.

All seeds are batched into a single (S*T, J) tensor for FK, so the GPU
processes all seeds in parallel rather than sequentially.

Collision costs use a one-sided exponential barrier: zero when safe
(sd > margin), steep exponential when near or inside obstacles.  When a
table plane is set, the trajectory is initialized as a parabolic arch
(ramped deltas) that keeps early waypoints near the start configuration,
preventing the optimizer from cutting through the table.
"""

import time

import torch
import pytorch_kinematics as pk

from .collision_model import (
    CapsuleCollisionModel,
    SELF_COLLISION_PAIRS,
    VoxelSDF,
    capsule_capsule_distance,
    capsule_halfplane_distance,
)


class TrajectoryOptimizer:
    """Gradient-based trajectory planner with collision avoidance."""

    def __init__(
        self,
        chain: pk.Chain,
        capsule_model: CapsuleCollisionModel,
        joint_limits: tuple[list[float], list[float]],
        device: str | torch.device = "cuda",
        collision_alpha: float = 50.0,
        max_joint_step: float = 0.1,
    ):
        self.device = torch.device(device)
        self.chain = chain.to(dtype=torch.float32, device=str(self.device))
        self.capsules = capsule_model
        self.lower = torch.tensor(
            joint_limits[0], dtype=torch.float32, device=self.device
        )
        self.upper = torch.tensor(
            joint_limits[1], dtype=torch.float32, device=self.device
        )
        self.collision_alpha = collision_alpha
        self.max_step_default = max_joint_step
        self.table_plane: tuple[float, tuple[float, float, float, float]] | None = None
        self.table_polygon: torch.Tensor | None = None

        # Cost weights.  Cartesian EE path length is the main objective so
        # more iterations always → shorter end-effector path.  Joint-space
        # smoothness prevents jerky motion.  Collision barriers use the same
        # margin as the post-hoc safety check (SAFETY_MARGIN) to avoid
        # routing close to obstacles that later fail validation.
        self.w_cart_path = 5000.0  # dominant — Cartesian EE path length
        self.w_smooth = 1.0        # joint-space acceleration penalty
        self.w_env = 20.0          # obstacle avoidance
        self.w_self = 5.0          # self-collision
        self.w_table = 100.0       # table backup (lift phase is primary)
        self.w_goal_residual = 10.0

        # Store the EE link name for Cartesian path cost
        self._ee_link = chain.get_link_names()[-1]

    def set_table(
        self,
        table_plane: tuple[float, tuple[float, float, float, float]],
        table_polygon: torch.Tensor | None = None,
    ):
        """Set the table collision plane and optional polygon.

        Use :func:`pointcloud.compute_table_plane` to compute both values
        from camera intrinsics and a point cloud.  This ensures the
        visualisation and the optimiser use the same table region.

        Args:
            table_plane: ``(plane_z, (x_min, x_max, y_min, y_max))``
            table_polygon: ``(V, 2)`` polygon vertices on the device, or None
                to fall back to the AABB bounds.
        """
        self.table_plane = table_plane
        self.table_polygon = table_polygon

    def _q_to_phi_local(self, q: torch.Tensor) -> torch.Tensor:
        """Map joint angles to unconstrained space via inverse-sigmoid.

        Used only by repair_table_violations for local phi-space optimisation.
        """
        t = (q - self.lower) / (self.upper - self.lower)
        t = t.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(t / (1.0 - t))

    def _phi_to_q_local(self, phi: torch.Tensor) -> torch.Tensor:
        """Map unconstrained parameters back to joint angles within limits.

        Used only by repair_table_violations for local phi-space optimisation.
        """
        return self.lower + (self.upper - self.lower) * torch.sigmoid(phi)

    @staticmethod
    def _soft_clamp(
        q: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        margin: float = 0.05,
    ) -> torch.Tensor:
        """Differentiable joint limit enforcement using softplus.

        Smoothly pushes values inside [lower, upper] with a transition zone
        of *margin* radians (~3°).  Gradients flow everywhere — no dead zones.
        """
        beta = 1.0 / margin
        q = lower + torch.nn.functional.softplus(q - lower, beta=beta)
        q = upper - torch.nn.functional.softplus(upper - q, beta=beta)
        return q

    @staticmethod
    def _one_sided_barrier(
        sd: torch.Tensor, alpha: float, margin: float = 0.02, max_exp: float = 5.0,
    ) -> torch.Tensor:
        """Zero when safe (sd > margin), steep exponential when near/inside collision.

        Beyond the exponential zone (penetration > max_exp/alpha), the cost
        extends linearly so the gradient never vanishes for deep penetrations.
        """
        penetration = torch.relu(margin - sd)
        exp_limit = max_exp / alpha
        exp_zone = penetration.clamp(max=exp_limit)
        overshoot = penetration - exp_zone  # > 0 for deep penetrations
        # exp(alpha * exp_limit) = exp(max_exp), derivative at boundary = alpha * exp(max_exp)
        exp_cost = torch.exp(alpha * exp_zone) - 1.0
        # Linear extension with same slope as the exponential at the boundary
        linear_cost = overshoot * alpha * exp_cost.clamp(min=1.0).detach()
        return exp_cost + linear_cost

    def _compute_max_step(
        self, q_start: torch.Tensor, q_goal: torch.Tensor, T: int
    ) -> torch.Tensor:
        """Per-joint max step size, auto-enlarged for large motions.

        Returns (J,) tensor.  If any joint requires a per-step delta larger
        than ``self.max_step_default`` to reach the goal in T-1 steps, that
        joint's limit is enlarged with a 20% margin and a warning is logged.
        """
        required = (q_goal - q_start).abs() / (T - 1)  # (J,)
        max_step = torch.full_like(required, self.max_step_default)
        # Ensure at least 2x headroom over required — keeps the initial tanh
        # input at ≤atanh(0.5)≈0.55 so gradients flow well (tanh'≥0.75).
        min_step = required * 2.0
        needs_enlarge = min_step > max_step
        if needs_enlarge.any():
            max_step[needs_enlarge] = min_step[needs_enlarge]
            joints = needs_enlarge.nonzero(as_tuple=True)[0].tolist()
            print(
                f"[trajectory-opt] Auto-enlarged max_step for joints {joints}: "
                f"{min_step[needs_enlarge].cpu().tolist()}"
            )
        return max_step

    def _delta_logits_to_trajectory(
        self,
        delta_logits: torch.Tensor,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        max_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert delta logits to a full trajectory with bridge correction.

        Args:
            delta_logits: (S, T-1, J) unconstrained parameters.
            q_start: (J,) start configuration.
            q_goal: (J,) goal configuration.
            max_step: (J,) per-joint max step size.

        Returns:
            q_traj: (S, T, J) trajectory with q[0]=q_start, q[-1]≈q_goal.
            goal_residual: (S, J) raw cumsum error before bridge correction.
        """
        S = delta_logits.shape[0]
        T_minus_1 = delta_logits.shape[1]
        J = delta_logits.shape[2]

        # Bounded deltas via tanh
        delta_q = max_step * torch.tanh(delta_logits)  # (S, T-1, J)

        # Cumulative sum from start
        q_cumsum = q_start + torch.cumsum(delta_q, dim=1)  # (S, T-1, J)

        # Bridge correction: linearly distribute residual so q[-1] = q_goal exactly
        goal_residual = q_goal - q_cumsum[:, -1, :]  # (S, J)
        t_frac = torch.linspace(
            1.0 / T_minus_1, 1.0, T_minus_1, device=self.device
        ).view(1, T_minus_1, 1)  # (1, T-1, 1)
        q_corrected = q_cumsum + t_frac * goal_residual.unsqueeze(1)  # (S, T-1, J)

        # Soft-clamp interior waypoints to joint limits (endpoints are already valid)
        q_interior = self._soft_clamp(
            q_corrected[:, :-1], self.lower, self.upper
        )  # (S, T-2, J)

        # Assemble full trajectory with exact start and goal
        q_start_exp = q_start.view(1, 1, J).expand(S, 1, J)
        q_goal_exp = q_goal.view(1, 1, J).expand(S, 1, J)
        q_traj = torch.cat([q_start_exp, q_interior, q_goal_exp], dim=1)  # (S, T, J)

        return q_traj, goal_residual

    def optimize(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        point_cloud: torch.Tensor | None = None,
        T: int = 200,
        num_seeds: int = 8,
        max_iters: int = 200,
        lr: float = 0.01,
        patience: int = 50,
    ) -> tuple[torch.Tensor, float]:
        """Multi-start GPU-batched trajectory optimisation.

        Parameterises the trajectory as bounded delta joint angles per step,
        each passed through tanh with a per-joint max step size.  This
        naturally guarantees bounded velocity and prevents loops.

        Args:
            q_start: (J,) start joint configuration.
            q_goal: (J,) goal joint configuration.
            point_cloud: (N, 3) obstacle points in robot base frame, or None.
            T: Number of waypoints.
            num_seeds: Number of random initialisations (batched on GPU).
            max_iters: Adam iterations.
            lr: Learning rate.
            patience: Stop early if no seed improves by >1% for this many iters.

        Returns:
            (best_traj, best_cost) — best_traj is (T, J) tensor.
        """
        q_start = q_start.to(self.device).detach().clamp(self.lower, self.upper)
        q_goal = q_goal.to(self.device).detach().clamp(self.lower, self.upper)
        S = num_seeds
        J = q_start.shape[0]

        sdf = None
        if point_cloud is not None and len(point_cloud) > 0:
            sdf = VoxelSDF(point_cloud.to(self.device).detach())

        # Compute per-joint max step (auto-enlarged for large motions)
        max_step = self._compute_max_step(q_start, q_goal, T)  # (J,)

        # Initialise delta_logits so that the trajectory starts as a linear
        # interpolation: nominal_delta = (q_goal - q_start) / (T-1), then
        # invert through tanh to get the logit.
        nominal_delta = (q_goal - q_start) / (T - 1)  # (J,)
        ratio = (nominal_delta / max_step).clamp(-0.99, 0.99)
        logit_init = torch.atanh(ratio)  # (J,)

        # Expand to (S, T-1, J) — all seeds start from the same linear interp
        delta_logits = logit_init.view(1, 1, J).expand(S, T - 1, J).clone()

        # Seed diversity: add noise to seeds 1..S-1
        if S > 1:
            delta_logits[1:] += torch.randn(S - 1, T - 1, J, device=self.device) * 0.3

        delta_logits = delta_logits.requires_grad_(True)
        optimizer = torch.optim.Adam([delta_logits], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iters, eta_min=lr * 0.01
        )

        best_seed_costs = torch.full((S,), float("inf"), device=self.device)
        stale_counts = torch.zeros(S, dtype=torch.int64, device=self.device)
        t0 = time.perf_counter()

        for it in range(max_iters):
            optimizer.zero_grad()

            q_traj, goal_residual = self._delta_logits_to_trajectory(
                delta_logits, q_start, q_goal, max_step
            )  # (S, T, J), (S, J)

            costs = self._total_cost_batched(
                q_traj, sdf, S, T, goal_residual=goal_residual
            )  # (S,)
            costs.sum().backward()
            optimizer.step()
            scheduler.step()

            # Per-seed early stopping
            with torch.no_grad():
                improved = costs < best_seed_costs * 0.99
                best_seed_costs = torch.where(improved, costs.detach(), best_seed_costs)
                stale_counts = torch.where(
                    improved, torch.zeros_like(stale_counts), stale_counts + 1
                )
                if (stale_counts >= patience).all():
                    break

        # Select best seed
        with torch.no_grad():
            q_traj, goal_residual = self._delta_logits_to_trajectory(
                delta_logits, q_start, q_goal, max_step
            )
            final_costs = self._total_cost_batched(
                q_traj, sdf, S, T, goal_residual=goal_residual
            )
            best_idx = final_costs.argmin()
            best_traj = q_traj[best_idx].cpu()
            best_cost = final_costs[best_idx].item()

        elapsed = time.perf_counter() - t0
        print(
            f"[trajectory-opt] {S} seeds × {it + 1} iters in "
            f"{elapsed:.2f}s (best cost={best_cost:.4f})"
        )
        for s in range(S):
            print(f"  seed {s}: cost={final_costs[s].item():.4f}")

        return best_traj, best_cost

    @staticmethod
    def _smooth_trajectory(
        traj: torch.Tensor,
        max_acc: float = 0.08,
        passes: int = 10,
    ) -> torch.Tensor:
        """Post-optimization smoothing: iteratively damp acceleration spikes."""
        traj = traj.clone()
        T = traj.shape[0]
        if T < 3:
            return traj
        for _ in range(passes):
            vel = traj[1:] - traj[:-1]
            acc = vel[1:] - vel[:-1]
            spike = acc.abs().max(dim=1).values > max_acc
            if not spike.any():
                break
            for i in range(T - 2):
                if spike[i]:
                    traj[i + 1] = (traj[i] + traj[i + 2]) / 2.0
        return traj

    # ------------------------------------------------------------------
    # Batched cost computation — all S seeds processed in one FK call
    # ------------------------------------------------------------------

    def _total_cost_batched(
        self,
        q_traj: torch.Tensor,
        sdf: VoxelSDF | None,
        S: int,
        T: int,
        goal_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-seed costs for (S, T, J) trajectory tensor.

        Args:
            goal_residual: (S, J) bridge error from _delta_logits_to_trajectory,
                or None for legacy single-trajectory calls.

        Returns (S,) cost vector.
        """
        J = q_traj.shape[-1]

        # Joint-space smoothness (acceleration)
        vel = q_traj[:, 1:] - q_traj[:, :-1]             # (S, T-1, J)
        acc = vel[:, 1:] - vel[:, :-1]                     # (S, T-2, J)
        smooth_cost = (acc ** 2).sum(dim=(-1, -2))          # (S,)

        # FK on flattened (S*T, J) — used for both collision and path cost
        q_flat = q_traj.reshape(S * T, J)
        transforms = self.chain.forward_kinematics(q_flat, end_only=False)

        # Cartesian EE path length (directly optimises what we measure)
        ee_tf = transforms[self._ee_link]
        ee_pos = ee_tf.get_matrix()[:, :3, 3]              # (S*T, 3)
        ee_pos = ee_pos.reshape(S, T, 3)
        ee_vel = ee_pos[:, 1:] - ee_pos[:, :-1]            # (S, T-1, 3)
        cart_path_cost = (ee_vel ** 2).sum(dim=(-1, -2))    # (S,)

        env_cost = torch.zeros(S, device=self.device)
        if sdf is not None:
            env_cost = self._env_cost_batched(transforms, sdf, S, T)

        self_cost = self._self_cost_batched(transforms, S, T)

        table_cost = torch.zeros(S, device=self.device)
        if self.table_plane is not None:
            table_cost = self._table_cost_batched(transforms, S, T)

        # Goal residual — encourages raw cumsum to reach goal
        residual_cost = torch.zeros(S, device=self.device)
        if goal_residual is not None:
            residual_cost = (goal_residual ** 2).sum(dim=-1)  # (S,)

        return (
            self.w_cart_path * cart_path_cost
            + self.w_smooth * smooth_cost
            + self.w_env * env_cost
            + self.w_self * self_cost
            + self.w_table * table_cost
            + self.w_goal_residual * residual_cost
        )

    def _env_cost_batched(
        self,
        transforms: dict,
        sdf: VoxelSDF,
        S: int,
        T: int,
    ) -> torch.Tensor:
        """SDF-based env collision cost, returns (S,)."""
        alpha = self.collision_alpha
        n_samples = 8
        ts = torch.linspace(0, 1, n_samples, device=self.device)
        total = torch.zeros(S, device=self.device)

        for link_name in self.capsules.link_names:
            if link_name not in transforms:
                continue
            p0_w, p1_w = self.capsules.capsule_endpoints_world(link_name, transforms)
            radius = self.capsules.radii[link_name]
            # p0_w: (S*T, 3), pts: (S*T, n_samples, 3)
            pts = p0_w.unsqueeze(1) + ts.view(1, -1, 1) * (p1_w - p0_w).unsqueeze(1)
            sd = sdf.query(pts) - radius  # (S*T, n_samples)
            barrier = self._one_sided_barrier(sd, alpha)  # (S*T, n_samples)
            # Sum over samples and timesteps per seed
            total = total + barrier.sum(dim=-1).reshape(S, T).sum(dim=-1)

        for link_name in self.capsules.box_link_names:
            parent = self.capsules.box_parent_link[link_name]
            if parent not in transforms:
                continue
            tf = transforms[parent]
            mat = tf.get_matrix()  # (S*T, 4, 4)
            R = mat[:, :3, :3]
            t = mat[:, :3, 3]
            center = self.capsules.box_centers[link_name]
            half_ext = self.capsules.box_half_extents[link_name]
            center_w = torch.einsum("bij,j->bi", R, center) + t  # (S*T, 3)
            box_radius = half_ext.norm().item()
            sd = sdf.query(center_w) - box_radius  # (S*T,)
            barrier = self._one_sided_barrier(sd, alpha)
            total = total + barrier.reshape(S, T).sum(dim=-1)

        return total

    def _self_cost_batched(
        self, transforms: dict, S: int, T: int
    ) -> torch.Tensor:
        """Self-collision cost, returns (S,)."""
        alpha = self.collision_alpha
        cost = torch.zeros(S, device=self.device)
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

            sd = capsule_capsule_distance(p0_i, p1_i, r_i, p0_j, p1_j, r_j)  # (S*T,)
            barrier = self._one_sided_barrier(sd, alpha)
            cost = cost + barrier.reshape(S, T).sum(dim=-1)

        return cost

    def _table_cost_batched(
        self, transforms: dict, S: int, T: int
    ) -> torch.Tensor:
        """Table half-plane collision cost, returns (S,).

        Checks both capsule links AND gripper boxes against the table.
        Uses a one-sided barrier: zero cost when above the table surface
        (plus margin), steep exponential penalty when near or below it.
        """
        table_alpha = max(self.collision_alpha * 4, 200.0)
        plane_z, bounds = self.table_plane
        total = torch.zeros(S, device=self.device)

        table_margin = 0.02  # 20mm — matches SAFETY_MARGIN

        for link_name in self.capsules.link_names:
            if link_name not in transforms:
                continue
            p0_w, p1_w = self.capsules.capsule_endpoints_world(link_name, transforms)
            radius = self.capsules.radii[link_name]
            sd = capsule_halfplane_distance(
                p0_w, p1_w, radius, plane_z, bounds,
                plane_polygon=self.table_polygon,
            )  # (S*T,)
            barrier = self._one_sided_barrier(sd, table_alpha, margin=table_margin)
            total = total + barrier.reshape(S, T).sum(dim=-1)

        # Gripper boxes — approximate as sphere (center + bounding radius)
        for link_name in self.capsules.box_link_names:
            parent = self.capsules.box_parent_link[link_name]
            if parent not in transforms:
                continue
            tf = transforms[parent]
            mat = tf.get_matrix()  # (S*T, 4, 4)
            R = mat[:, :3, :3]
            t = mat[:, :3, 3]
            center = self.capsules.box_centers[link_name]
            half_ext = self.capsules.box_half_extents[link_name]
            center_w = torch.einsum("bij,j->bi", R, center) + t  # (S*T, 3)
            box_radius = half_ext.norm().item()
            # Treat as a point capsule (p0=p1=center) with radius=box_radius
            sd = capsule_halfplane_distance(
                center_w, center_w, box_radius, plane_z, bounds,
                plane_polygon=self.table_polygon,
            )  # (S*T,)
            barrier = self._one_sided_barrier(sd, table_alpha, margin=table_margin)
            total = total + barrier.reshape(S, T).sum(dim=-1)

        return total

    # ------------------------------------------------------------------
    # Post-optimization table violation repair
    # ------------------------------------------------------------------

    def repair_table_violations(
        self,
        traj: torch.Tensor,
        margin: float = 0.025,
        max_iters: int = 300,
        lr: float = 0.05,
    ) -> torch.Tensor:
        """Push violating waypoints above the table using gradient descent.

        Optimises joint angles for all violating interior waypoints
        simultaneously, using differentiable FK to compute a quadratic
        table-penetration penalty.  Smoothness is enforced via an
        acceleration cost computed *within* the optimised block (plus
        boundary terms to the fixed neighbours), so consecutive violating
        waypoints pull each other smoothly upward rather than fighting
        against stale neighbour values.

        Args:
            traj: (T, J) trajectory on CPU.
            margin: Clearance above table plane (metres).
            max_iters: Max gradient-descent steps.
            lr: Learning rate.

        Returns:
            Repaired (T, J) trajectory on CPU.
        """
        if self.table_plane is None:
            return traj

        plane_z = self.table_plane[0]
        target_z = plane_z + margin
        traj = traj.clone()
        traj_dev = traj.to(self.device)
        T, J = traj_dev.shape

        # Find violating interior waypoints — must be below table z
        # AND within the table's XY footprint (polygon or AABB).
        # Checks both capsule links and gripper boxes.
        bounds = self.table_plane[1] if self.table_plane else None
        with torch.no_grad():
            transforms = self.chain.forward_kinematics(traj_dev, end_only=False)
            violating = set()
            for link_name in self.capsules.link_names:
                if link_name not in transforms:
                    continue
                p0_w, p1_w = self.capsules.capsule_endpoints_world(
                    link_name, transforms
                )
                radius = self.capsules.radii[link_name]
                sd = capsule_halfplane_distance(
                    p0_w, p1_w, radius, plane_z, bounds,
                    plane_polygon=self.table_polygon,
                )  # (T,) — negative means below table within XY bounds
                for t in range(1, T - 1):
                    if sd[t] < 0:
                        violating.add(t)
            # Gripper boxes
            for box_name in self.capsules.box_link_names:
                parent = self.capsules.box_parent_link[box_name]
                if parent not in transforms:
                    continue
                tf = transforms[parent]
                mat = tf.get_matrix()
                R = mat[:, :3, :3]
                t_vec = mat[:, :3, 3]
                center = self.capsules.box_centers[box_name]
                half_ext = self.capsules.box_half_extents[box_name]
                center_w = torch.einsum("bij,j->bi", R, center) + t_vec
                box_radius = half_ext.norm().item()
                sd = capsule_halfplane_distance(
                    center_w, center_w, box_radius, plane_z, bounds,
                    plane_polygon=self.table_polygon,
                )
                for t in range(1, T - 1):
                    if sd[t] < 0:
                        violating.add(t)

        if not violating:
            return traj

        v_list = sorted(violating)
        v_idx = torch.tensor(v_list, dtype=torch.long, device=self.device)
        n_viol = len(v_list)
        print(
            f"  [table-repair] {n_viol} waypoints below table "
            f"(wp {v_list[0]}-{v_list[-1]}), running gradient repair..."
        )

        # Fixed boundary neighbours (not being optimised)
        q_left_fixed = traj_dev[v_list[0] - 1].detach()   # (J,)
        q_right_fixed = traj_dev[min(v_list[-1] + 1, T - 1)].detach()  # (J,)

        # Optimise in unconstrained phi space for joint limit compliance
        q_orig = traj_dev[v_idx].clone()  # (V, J)
        phi = self._q_to_phi_local(q_orig).clone().requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=lr)

        for it in range(max_iters):
            opt.zero_grad()

            q = self._phi_to_q_local(phi)  # (V, J)
            V = q.shape[0]

            # 1) Smoothness: acceleration within the block + boundaries.
            #    Build the extended sequence [left_fixed, q[0..V-1], right_fixed]
            #    then compute acc = q[i-1] + q[i+1] - 2*q[i] for all i.
            seq = torch.cat([
                q_left_fixed.unsqueeze(0), q, q_right_fixed.unsqueeze(0)
            ], dim=0)  # (V+2, J)
            acc = seq[:-2] + seq[2:] - 2.0 * seq[1:-1]  # (V, J)
            smooth_cost = acc.pow(2).sum()

            # 2) Table violation: differentiable through FK, XY-aware.
            #    Checks both capsule links and gripper boxes.
            transforms = self.chain.forward_kinematics(q, end_only=False)
            table_cost = torch.zeros(1, device=self.device)
            for link_name in self.capsules.link_names:
                if link_name not in transforms:
                    continue
                p0_w, p1_w = self.capsules.capsule_endpoints_world(
                    link_name, transforms
                )
                radius = self.capsules.radii[link_name]
                sd = capsule_halfplane_distance(
                    p0_w, p1_w, radius, plane_z, bounds,
                    plane_polygon=self.table_polygon,
                )  # (V,)
                violation = torch.relu(margin - sd)  # want sd >= margin
                table_cost = table_cost + violation.pow(2).sum()
            # Gripper boxes
            for box_name in self.capsules.box_link_names:
                parent = self.capsules.box_parent_link[box_name]
                if parent not in transforms:
                    continue
                tf = transforms[parent]
                mat = tf.get_matrix()
                R = mat[:, :3, :3]
                t_vec = mat[:, :3, 3]
                center = self.capsules.box_centers[box_name]
                half_ext = self.capsules.box_half_extents[box_name]
                center_w = torch.einsum("bij,j->bi", R, center) + t_vec
                box_radius = half_ext.norm().item()
                sd = capsule_halfplane_distance(
                    center_w, center_w, box_radius, plane_z, bounds,
                    plane_polygon=self.table_polygon,
                )
                violation = torch.relu(margin - sd)
                table_cost = table_cost + violation.pow(2).sum()

            loss = 5.0 * smooth_cost + 10000.0 * table_cost
            loss.backward()
            opt.step()

            if it % 50 == 0 or table_cost.item() < 1e-8:
                print(
                    f"    iter {it}: table_cost={table_cost.item():.6f} "
                    f"smooth={smooth_cost.item():.4f}"
                )
            if table_cost.item() < 1e-8:
                print(f"  [table-repair] Converged in {it + 1} iters")
                break

        # Apply repairs
        with torch.no_grad():
            q_repaired = self._phi_to_q_local(phi)
            traj_dev[v_idx] = q_repaired

            # Verify using XY-aware check (capsules + boxes)
            transforms = self.chain.forward_kinematics(traj_dev, end_only=False)
            remaining = 0
            for link_name in self.capsules.link_names:
                if link_name not in transforms:
                    continue
                p0_w, p1_w = self.capsules.capsule_endpoints_world(
                    link_name, transforms
                )
                radius = self.capsules.radii[link_name]
                sd = capsule_halfplane_distance(
                    p0_w, p1_w, radius, plane_z, bounds,
                    plane_polygon=self.table_polygon,
                )
                remaining += (sd[1:-1] < 0).sum().item()
            for box_name in self.capsules.box_link_names:
                parent = self.capsules.box_parent_link[box_name]
                if parent not in transforms:
                    continue
                tf = transforms[parent]
                mat = tf.get_matrix()
                R = mat[:, :3, :3]
                t_vec = mat[:, :3, 3]
                center = self.capsules.box_centers[box_name]
                half_ext = self.capsules.box_half_extents[box_name]
                center_w = torch.einsum("bij,j->bi", R, center) + t_vec
                box_radius = half_ext.norm().item()
                sd = capsule_halfplane_distance(
                    center_w, center_w, box_radius, plane_z, bounds,
                    plane_polygon=self.table_polygon,
                )
                remaining += (sd[1:-1] < 0).sum().item()
            if remaining > 0:
                print(
                    f"  [table-repair] WARNING: {remaining} link-waypoint "
                    f"violations remain after repair"
                )
            else:
                print("  [table-repair] All interior waypoints clear of table")

        return traj_dev.cpu()

    # ------------------------------------------------------------------
    # Legacy single-trajectory cost (used by validate_trajectory etc.)
    # ------------------------------------------------------------------

    def _total_cost(
        self, q_traj: torch.Tensor, sdf: VoxelSDF | None
    ) -> torch.Tensor:
        """Single-trajectory cost for (T, J) input. Returns scalar."""
        return self._total_cost_batched(
            q_traj.unsqueeze(0), sdf, S=1, T=q_traj.shape[0],
            goal_residual=None,
        ).squeeze(0)
