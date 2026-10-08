"""Universal Algorithmic Oracle for Parametric Franka Environments."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


# TCP height above the body origin at which the oracle closes the gripper (m). The parametric
# env attaches bodies within 6.5 cm of the TCP; descending lower lets the fingers shove wide parts.
GRASP_HEIGHT = 0.04


class ParametricOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for parametric tasks."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        target_goal = self.env.target_goal_pos.copy()

        if self.env.task_type in ("pick_place", "insert"):
            obj_pos = self.body_position(self.env.target_id)
            waypoints = [
                # 1. Hover above object
                np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.12, 1.0]),
                # 2. Descend to grasp height (within the env's grasp radius, clear of wide parts)
                np.array([obj_pos[0], obj_pos[1], obj_pos[2] + GRASP_HEIGHT, 1.0]),
                # 3. Clamp object
                np.array([obj_pos[0], obj_pos[1], obj_pos[2] + GRASP_HEIGHT, -1.0]),
                # 4. Lift object up, clearing any required minimum lift
                np.array([obj_pos[0], obj_pos[1], obj_pos[2] + max(0.15, self.env.min_lift + 0.04), -1.0]),
                # 5. Move above target goal
                np.array([target_goal[0], target_goal[1], target_goal[2] + 0.12, -1.0]),
                # 6. Lower to goal
                np.array([target_goal[0], target_goal[1], target_goal[2] + 0.015, -1.0]),
                # 7. Release object
                np.array([target_goal[0], target_goal[1], target_goal[2] + 0.015, 1.0]),
                # 8. Retract upward
                np.array([target_goal[0], target_goal[1], target_goal[2] + 0.15, 1.0]),
                # 9. Return to ready pose
                np.array([0.36, 0.0, 0.50, 1.0]),
            ]
        elif self.env.task_type == "press_button":
            waypoints = [
                # 1. Approach in front of button
                np.array([target_goal[0] - 0.08, target_goal[1], target_goal[2], 1.0]),
                # 2. Press forward onto button
                np.array([target_goal[0], target_goal[1], target_goal[2], -1.0]),
                # 3. Retract back
                np.array([target_goal[0] - 0.10, target_goal[1], target_goal[2], 1.0]),
                # 4. Return to ready pose
                np.array([0.36, 0.0, 0.50, 1.0]),
            ]
        else:
            raise NotImplementedError(f"Task type {self.env.task_type} waypoints not implemented.")

        return self.execute_waypoints(obs, waypoints)
