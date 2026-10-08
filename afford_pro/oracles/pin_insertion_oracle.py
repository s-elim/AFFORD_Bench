"""Local Algorithmic Oracle for Industrial Pin Insertion."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class PinInsertionOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for dowel pin insertion."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        pin_pos = self.body_position(self.env.pin_id)
        target_pos = self.env.target_hole_pos.copy()

        waypoints = [
            # 1. Hover above pin feeder cup
            np.array([pin_pos[0], pin_pos[1], pin_pos[2] + 0.12, 1.0]),
            # 2. Descend onto pin head
            np.array([pin_pos[0], pin_pos[1], pin_pos[2] + 0.012, 1.0]),
            # 3. Clamp pin
            np.array([pin_pos[0], pin_pos[1], pin_pos[2] + 0.012, -1.0]),
            # 4. Lift pin up
            np.array([pin_pos[0], pin_pos[1], pin_pos[2] + 0.15, -1.0]),
            # 5. Move above fixture locating hole
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.12, -1.0]),
            # 6. Insert pin into hole
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, -1.0]),
            # 7. Release pin
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, 1.0]),
            # 8. Retract upward
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
