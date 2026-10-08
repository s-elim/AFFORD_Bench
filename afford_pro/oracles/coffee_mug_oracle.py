"""Local Algorithmic Oracle for Coffee Mug to Espresso."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class CoffeeMugOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for coffee mug placement."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        mug_pos = self.body_position(self.env.mug_id)
        target_pos = self.env.target_drip_tray_pos.copy()

        waypoints = [
            # 1. Hover above mug in bin
            np.array([mug_pos[0], mug_pos[1], mug_pos[2] + 0.12, 1.0]),
            # 2. Descend onto mug rim
            np.array([mug_pos[0], mug_pos[1], mug_pos[2] + 0.015, 1.0]),
            # 3. Clamp mug
            np.array([mug_pos[0], mug_pos[1], mug_pos[2] + 0.015, -1.0]),
            # 4. Lift mug
            np.array([mug_pos[0], mug_pos[1], mug_pos[2] + 0.16, -1.0]),
            # 5. Move in front of espresso drip tray
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.12, -1.0]),
            # 6. Lower onto drip tray
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, -1.0]),
            # 7. Release mug
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.015, 1.0]),
            # 8. Retract outward
            np.array([target_pos[0] - 0.08, target_pos[1], target_pos[2] + 0.12, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
