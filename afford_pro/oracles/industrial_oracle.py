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

        recorded_obs: dict[str, list[np.ndarray]] = {k: [v] for k, v in obs.items()}
        recorded_actions: list[np.ndarray] = []
        recorded_rewards: list[float] = [0.0]
        recorded_dones: list[bool] = [False]

        # Target coordinates dynamically queried from current environment state
        shaft_pos = self.env.data.xpos[self.env.shaft_id].copy()
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

        # Execute waypoints with smooth S-curve interpolation
        step_budget_per_segment = 24  # steps per movement segment
        for wp in waypoints:
            target_pos = wp[:3]
            target_grip = wp[3]

            cur_pos = self.env.tcp_position.copy()
            dist = np.linalg.norm(target_pos - cur_pos)
            steps = max(6, int(step_budget_per_segment * max(0.5, dist / 0.10)))

            for t in np.linspace(0, 1, steps):
                # Cubic smoothstep: 3t^2 - 2t^3
                alpha = t * t * (3.0 - 2.0 * t)
                interp_pos = (1.0 - alpha) * cur_pos + alpha * target_pos
                delta_xyz = (interp_pos - self.env.tcp_position) / self.env.action_scale
                delta_xyz = np.clip(delta_xyz, -1.0, 1.0)

                # Action: [dx, dy, dz, 0, 0, 0, gripper]
                action = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2], 0.0, 0.0, 0.0, target_grip], dtype=np.float32)

                obs, reward, terminated, truncated, info = self.env.step(action)

                recorded_actions.append(action)
                for k, v in obs.items():
                    recorded_obs[k].append(v)
                recorded_rewards.append(reward)
                recorded_dones.append(terminated)

                if terminated:
                    break
            if self.env.check_success():
                break

        # Trim final observation list so lengths align with actions (T steps)
        for k in recorded_obs:
            recorded_obs[k] = recorded_obs[k][:len(recorded_actions)]
        recorded_rewards = recorded_rewards[:len(recorded_actions)]
        recorded_dones = recorded_dones[:len(recorded_actions)]

        return recorded_actions, recorded_obs, recorded_rewards, recorded_dones
