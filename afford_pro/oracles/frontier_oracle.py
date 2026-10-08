"""Frontier manipulation oracle for Domain 8 tasks."""
from __future__ import annotations

import numpy as np
from afford_pro.oracles.parametric_oracle import ParametricOracle, OPEN, CLOSE


class FrontierOracle(ParametricOracle):
    """Oracle supporting reactive perturbations and non-prehensile pushing phases."""

    def run(self) -> None:
        min_push = getattr(self.env, "min_push_distance", 0.0)
        frontier_type = getattr(self.env, "frontier_task_type", "")
        if min_push > 0.0 and frontier_type == "push_and_grasp":
            self._execute_push_phase(min_push)

        super().run()

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
