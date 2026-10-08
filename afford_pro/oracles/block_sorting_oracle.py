"""Local Algorithmic Oracle for Tabletop Block Sorting."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class BlockSortingOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for block sorting."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        obj_pos = self.body_position(self.env.target_id)
        target_pos = self.env.target_bin_pos.copy()

        waypoints = [
            # 1. Hover above target block
            np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.12, 1.0]),
            # 2. Descend onto block
            np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.010, 1.0]),
            # 3. Clamp block
            np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.010, -1.0]),
            # 4. Lift block up
            np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.16, -1.0]),
            # 5. Move above target bin
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.12, -1.0]),
            # 6. Lower into bin
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, -1.0]),
            # 7. Release block
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, 1.0]),
            # 8. Retract upward
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
