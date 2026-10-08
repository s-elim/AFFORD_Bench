"""Base Algorithmic Demonstration Oracle for Franka Environments."""
from __future__ import annotations

from typing import Any
import numpy as np

Episode = tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]


class BaseOracle:
    """Scripted waypoint oracle: plans from the settled reset state and tracks with the env's IK."""

    # Interpolation budget: env steps per 10 cm of TCP travel (12 steps = 0.6 s at 20 Hz).
    steps_per_10cm: int = 12

    def __init__(self, env: Any):
        self.env = env

    def generate_episode(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Episode:
        """Generate a complete successful episode rollout."""
        raise NotImplementedError

    def body_position(self, body_id: int) -> np.ndarray:
        """Current world position of a body; after env.reset() this is the settled pose."""
        return self.env.data.xpos[body_id].copy()

    def execute_waypoints(self, first_obs: dict[str, np.ndarray], waypoints: list[np.ndarray]) -> Episode:
        """Track [x, y, z, gripper] waypoints with smoothstep interpolation until success.

        Returns (actions, observations, rewards, dones) aligned to T recorded steps.
        """
        recorded_obs: dict[str, list[np.ndarray]] = {k: [v] for k, v in first_obs.items()}
        recorded_actions: list[np.ndarray] = []
        recorded_rewards: list[float] = [0.0]
        recorded_dones: list[bool] = [False]

        for wp in waypoints:
            target_pos, target_grip = wp[:3], wp[3]
            cur_pos = self.env.tcp_position.copy()
            dist = np.linalg.norm(target_pos - cur_pos)
            steps = max(6, int(self.steps_per_10cm * max(0.5, dist / 0.10)))

            terminated = False
            for t in np.linspace(0, 1, steps):
                # Cubic smoothstep: 3t^2 - 2t^3
                alpha = t * t * (3.0 - 2.0 * t)
                interp_pos = (1.0 - alpha) * cur_pos + alpha * target_pos
                delta_xyz = np.clip((interp_pos - self.env.tcp_position) / self.env.action_scale, -1.0, 1.0)

                # Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                action = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2], 0.0, 0.0, 0.0, target_grip], dtype=np.float32)
                obs, reward, terminated, _, _ = self.env.step(action)

                recorded_actions.append(action)
                for k, v in obs.items():
                    recorded_obs[k].append(v)
                recorded_rewards.append(reward)
                recorded_dones.append(terminated)
                if terminated:
                    break
            if terminated:
                break

        # Trim so observation, reward, and done lists align with the T actions.
        T = len(recorded_actions)
        return (
            recorded_actions,
            {k: v[:T] for k, v in recorded_obs.items()},
            recorded_rewards[:T],
            recorded_dones[:T],
        )
