"""Local Algorithmic Oracle for Culinary Cheese Placement."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class CheesePlaceOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for cheese placement."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        cheese_pos = self.body_position(self.env.cheese_id)
        target_pos = self.env.target_cheese_pos.copy()

        waypoints = [
            # 1. Hover above cheese on topping tray
            np.array([cheese_pos[0], cheese_pos[1], cheese_pos[2] + 0.12, 1.0]),
            # 2. Descend onto cheese
            np.array([cheese_pos[0], cheese_pos[1], cheese_pos[2] + 0.008, 1.0]),
            # 3. Clamp cheese
            np.array([cheese_pos[0], cheese_pos[1], cheese_pos[2] + 0.008, -1.0]),
            # 4. Lift cheese
            np.array([cheese_pos[0], cheese_pos[1], cheese_pos[2] + 0.14, -1.0]),
            # 5. Move above patty on assembly board
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.12, -1.0]),
            # 6. Lower onto patty
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.012, -1.0]),
            # 7. Release cheese
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.012, 1.0]),
            # 8. Retract upward
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
