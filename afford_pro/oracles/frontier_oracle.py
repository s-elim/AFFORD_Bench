"""Frontier manipulation oracle for Domain 8 tasks."""
from __future__ import annotations

import numpy as np
from afford_pro.oracles.parametric_oracle import (
    ParametricOracle,
    GraspLost,
    OPEN,
    CLOSE,
    CARRY_CLEARANCE,
    CARRY_MAX_STEP,
    CARRY_ACCEL_STEP,
    SETTLE_STEPS,
    JAM_RETRIES,
)


class FrontierOracle(ParametricOracle):
    """Oracle supporting reactive perturbations, goal shifts, and non-prehensile pushing phases."""

    def run(self) -> None:
        frontier_type = getattr(self.env, "frontier_task_type", "")
        min_push = getattr(self.env, "min_push_distance", 0.0)

        if frontier_type == "push_overhang":
            self._execute_push_to_goal()
            return

        if min_push > 0.0 and frontier_type in ("push_and_grasp", "push_mug_and_grasp"):
            self._execute_push_phase(min_push)

        super().run()

    def _execute_push_to_goal(self) -> None:
        """Push the object gently until target overhang distance is reached, then retract and settle."""
        env = self.env
        goal = env.goal_position
        obj_pos = self._obj()
        direction = goal[:2] - obj_pos[:2]
        d_norm = np.linalg.norm(direction)
        push_dir = direction / d_norm if d_norm > 1e-4 else np.array([0.0, 1.0])

        push_start = np.array([obj_pos[0] - 0.035 * push_dir[0], obj_pos[1] - 0.035 * push_dir[1], obj_pos[2] + 0.005])
        push_dist = getattr(env, "min_push_distance", 0.04) + 0.006
        push_end = np.array([obj_pos[0] + push_dist * push_dir[0], obj_pos[1] + push_dist * push_dir[1], obj_pos[2] + 0.005])

        # Approach behind object
        self.servo(push_start + np.array([0, 0, 0.08]), 0.0, CLOSE, tol=0.005, max_steps=40)
        # Lower to push depth
        self.servo(push_start, 0.0, CLOSE, tol=0.005, max_steps=40)
        # Push through object quasi-statically
        self.servo(push_end, 0.0, CLOSE, tol=0.005, max_steps=80, max_step=0.002)
        # Retract and open gripper
        self.servo(env.tcp_position + np.array([0, 0, 0.08]), 0.0, OPEN, tol=0.005, max_steps=30)
        # Wait for object to settle
        dof = env.free_qpos_to_dof.get(env.target_qposadr)
        for _ in range(50):
            if dof is not None and np.linalg.norm(env.data.qvel[dof : dof + 3]) < 0.01:
                break
            env.step(np.array([0, 0, 0, 0, 0, 0, OPEN]))

    def _execute_push_phase(self, min_push: float) -> None:
        """Push the object along the table surface before picking and placing."""
        env = self.env
        obj_pos = self._obj()
        goal = env.goal_position
        direction = goal[:2] - obj_pos[:2]
        d_norm = np.linalg.norm(direction)
        push_dir = direction / d_norm if d_norm > 1e-4 else np.array([0.0, -1.0])

        push_start = np.array([obj_pos[0] - 0.05 * push_dir[0], obj_pos[1] - 0.05 * push_dir[1], obj_pos[2] + 0.012])
        push_end = np.array([obj_pos[0] + (min_push + 0.02) * push_dir[0], obj_pos[1] + (min_push + 0.02) * push_dir[1], obj_pos[2] + 0.012])

        # Approach behind object
        self.servo(push_start + np.array([0, 0, 0.08]), 0.0, CLOSE, tol=0.005, max_steps=40)
        # Lower to push depth
        self.servo(push_start, 0.0, CLOSE, tol=0.005, max_steps=40)
        # Push through object along push vector
        self.servo(push_end, 0.0, CLOSE, tol=0.008, max_steps=60, max_step=0.006)
        # Retract and open gripper
        self.servo(push_end + np.array([0, 0, 0.08]), 0.0, OPEN, tol=0.005, max_steps=30)

    def _place(self) -> None:
        """Place with dynamic goal position tracking to handle fixture shifts."""
        env = self.env
        goal = env.goal_position
        offset = env.tcp_position - self._obj()
        carry_z = max(env.tcp_position[2], goal[2] + offset[2]) + CARRY_CLEARANCE
        carry_z = max(carry_z, env.start_z + env.min_lift + offset[2] + 0.01)
        carry_z = max(carry_z, self._path_clear_z(goal))
        yaw = self._place_yaw(goal + offset, carry_z)

        free_accel, self.accel_step = self.accel_step, CARRY_ACCEL_STEP
        try:
            self._carry_dynamic(carry_z, yaw)
        finally:
            self.accel_step = free_accel

    def _carry_dynamic(self, carry_z: float, yaw: float) -> None:
        """Carry the body while dynamically tracking env.goal_position."""
        env = self.env
        self.servo(np.array([*env.tcp_position[:2], carry_z]), env.tcp_yaw, CLOSE, tol=0.005, max_steps=80,
                   max_step=CARRY_MAX_STEP)
        misses = [0]

        def lost() -> bool:
            misses[0] = 0 if env.is_grasped(env.target_id) else misses[0] + 1
            if misses[0] >= 3:
                raise GraspLost
            return False

        def tcp_for(obj_target: np.ndarray) -> np.ndarray:
            return env.tcp_position + (obj_target - self._obj())

        def target_above() -> np.ndarray:
            g = env.goal_position
            return tcp_for(np.array([g[0], g[1], g[2] + 0.05]))

        # Dynamic servo towards goal xy at carry height
        self.servo(lambda: np.array([*target_above()[:2], carry_z]), yaw, CLOSE, tol=0.003, max_steps=160,
                   max_step=CARRY_MAX_STEP, until=lost)
        # Dynamic servo to hover above goal
        self.servo(target_above, yaw, CLOSE, tol=0.0008, max_steps=80, max_step=0.01, until=lost)

        # Hover until settled
        for _ in range(SETTLE_STEPS):
            lin, _ = env.object_speed()
            g = env.goal_position
            if lin < 0.005 and np.linalg.norm(self._obj()[:2] - g[:2]) < 0.0005:
                break
            self.act(self.action_towards(target_above(), yaw, CLOSE, max_step=0.002))
            lost()

        curr_goal = env.goal_position.copy()
        def supported() -> bool:
            contacts = env.body_contacts(env.target_id)
            return bool(contacts - env.finger_body_ids - env.robot_body_ids) and self._obj()[2] < curr_goal[2] + 0.004

        for attempt in range(JAM_RETRIES + 1):
            if self._descend(curr_goal, yaw, tcp_for, supported) or attempt == JAM_RETRIES:
                break
            entry = curr_goal + np.array([0.0, 0.0, 0.012])
            self.servo(lambda: tcp_for(entry), yaw, CLOSE, tol=0.0005, max_steps=40, max_step=0.004, until=lost)

        hold_tcp = env.tcp_position.copy()
        for _ in range(SETTLE_STEPS):
            lin, ang = env.object_speed()
            if lin < 0.005 and ang < 0.05:
                break
            self.act(self.action_towards(hold_tcp, yaw, CLOSE, max_step=0.002))
