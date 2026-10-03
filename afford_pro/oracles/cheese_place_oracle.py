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

        recorded_obs: dict[str, list[np.ndarray]] = {k: [v] for k, v in obs.items()}
        recorded_actions: list[np.ndarray] = []
        recorded_rewards: list[float] = [0.0]
        recorded_dones: list[bool] = [False]

        cheese_pos = self.env.data.xpos[self.env.cheese_id].copy()
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

        step_budget_per_segment = 24
        for wp in waypoints:
            target_p = wp[:3]
            target_grip = wp[3]

            cur_pos = self.env.tcp_position.copy()
            dist = np.linalg.norm(target_p - cur_pos)
            steps = max(6, int(step_budget_per_segment * max(0.5, dist / 0.10)))

            for t in np.linspace(0, 1, steps):
                alpha = t * t * (3.0 - 2.0 * t)
                interp_pos = (1.0 - alpha) * cur_pos + alpha * target_p
                delta_xyz = (interp_pos - self.env.tcp_position) / self.env.action_scale
                delta_xyz = np.clip(delta_xyz, -1.0, 1.0)

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

        for k in recorded_obs:
            recorded_obs[k] = recorded_obs[k][:len(recorded_actions)]
        recorded_rewards = recorded_rewards[:len(recorded_actions)]
        recorded_dones = recorded_dones[:len(recorded_actions)]

        return recorded_actions, recorded_obs, recorded_rewards, recorded_dones
