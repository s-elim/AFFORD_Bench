"""Local Algorithmic Oracle for Industrial Gear Meshing."""
from __future__ import annotations

from typing import Any
import numpy as np

from afford_pro.oracles.base_oracle import BaseOracle


class GearMeshOracle(BaseOracle):
    """100% local, zero-cost deterministic oracle solver for gear meshing."""

    def generate_episode(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        obs, _ = self.env.reset(seed=seed, options=options)

        # Target coordinates dynamically queried from current environment state
        gear_pos = self.body_position(self.env.gear_id)
        target_pos = self.env.target_mesh_pos.copy()

        # Build sequence of Cartesian waypoints [x, y, z, gripper_cmd]
        waypoints = [
            # 1. Move above spur gear
            np.array([gear_pos[0], gear_pos[1], gear_pos[2] + 0.12, 1.0]),
            # 2. Descend onto gear rim
            np.array([gear_pos[0], gear_pos[1], gear_pos[2] + 0.015, 1.0]),
            # 3. Clamp gear
            np.array([gear_pos[0], gear_pos[1], gear_pos[2] + 0.015, -1.0]),
            # 4. Lift gear up over feeder spindle
            np.array([gear_pos[0], gear_pos[1], gear_pos[2] + 0.16, -1.0]),
            # 5. Move above shaft seated in jig bore
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.14, -1.0]),
            # 6. Lower gear over shaft down into mesh
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.02, -1.0]),
            # 7. Release gear
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.02, 1.0]),
            # 8. Retract upward
            np.array([target_pos[0], target_pos[1], target_pos[2] + 0.16, 1.0]),
            # 9. Return to ready pose
            np.array([0.36, 0.0, 0.50, 1.0]),
        ]

        return self.execute_waypoints(obs, waypoints)
