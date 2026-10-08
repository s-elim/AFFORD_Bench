"""Local Algorithmic Oracle for Culinary Patty Stacking."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class PattyStackOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for patty stacking."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        patty_pos = self.body_position(self.env.patty_id)
        target_pos = self.env.target_stack_pos.copy()

        waypoints = [
            # 1. Hover above patty on grill
            np.array([patty_pos[0], patty_pos[1], patty_pos[2] + 0.12, 1.0]),
            # 2. Descend onto patty
            np.array([patty_pos[0], patty_pos[1], patty_pos[2] + 0.012, 1.0]),
            # 3. Clamp patty
            np.array([patty_pos[0], patty_pos[1], patty_pos[2] + 0.012, -1.0]),
            # 4. Lift patty
            np.array([patty_pos[0], patty_pos[1], patty_pos[2] + 0.15, -1.0]),
            # 5. Move above bottom bun on assembly board
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.12, -1.0]),
            # 6. Lower onto bun
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, -1.0]),
            # 7. Release patty
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, 1.0]),
            # 8. Retract upward
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
