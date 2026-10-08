"""Local Algorithmic Oracle for Industrial Shaft Assembly."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class ShaftAssemblyOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for shaft assembly."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        # Target coordinates dynamically queried from current environment state
        shaft_pos = self.body_position(self.env.shaft_id)
        jig_pos = self.env.jig_bore_pos.copy()

        # Build sequence of Cartesian waypoints [x, y, z, gripper_cmd]
        # Gripper: +1.0 = open, -1.0 = closed
        waypoints = [
            # 1. Move above shaft
            np.array([shaft_pos[0], shaft_pos[1], shaft_pos[2] + 0.12, 1.0]),
            # 2. Descend onto shaft mid-height
            np.array([shaft_pos[0], shaft_pos[1], shaft_pos[2] + 0.015, 1.0]),
            # 3. Clamp shaft
            np.array([shaft_pos[0], shaft_pos[1], shaft_pos[2] + 0.015, -1.0]),
            # 4. Lift shaft up
            np.array([shaft_pos[0], shaft_pos[1], shaft_pos[2] + 0.18, -1.0]),
            # 5. Move above jig bore
            np.array([jig_pos[0], jig_pos[1], jig_pos[2] + 0.15, -1.0]),
            # 6. Lower into bore
            np.array([jig_pos[0], jig_pos[1], jig_pos[2] + 0.02, -1.0]),
            # 7. Release shaft
            np.array([jig_pos[0], jig_pos[1], jig_pos[2] + 0.02, 1.0]),
            # 8. Retract upward
            np.array([jig_pos[0], jig_pos[1], jig_pos[2] + 0.18, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
