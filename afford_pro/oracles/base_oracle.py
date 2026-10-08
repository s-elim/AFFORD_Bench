"""Base demonstration oracle: episode recording and closed-loop TCP servoing."""
from __future__ import annotations

from typing import Any, Callable
import numpy as np

Episode = tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]
Target = np.ndarray | Callable[[], np.ndarray]


class EpisodeDone(Exception):
    """Raised when the env reports success or the oracle's step budget is spent."""


class BaseOracle:
    """Privileged scripted expert: reads simulator state every step and issues env actions."""

    max_steps: int = 600
    # Trapezoidal TCP profile: commanded displacement changes by at most this much per step (m),
    # i.e. 1.6 m/s^2 at 20 Hz, so carried parts are not shaken out of the grasp.
    accel_step: float = 0.004

    def __init__(self, env: Any):
        self.env = env

    def generate_episode(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Episode:
        """Generate one rollout; returns (actions, observations, rewards, dones) aligned to T steps."""
        obs, _ = self.env.reset(seed=seed, options=options)
        self._obs: dict[str, list[np.ndarray]] = {k: [v] for k, v in obs.items()}
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = [0.0]
        self._dones: list[bool] = [False]
        self._prev_delta = np.zeros(3)
        try:
            self.run()
        except EpisodeDone:
            pass
        T = len(self._actions)
        return self._actions, {k: v[:T] for k, v in self._obs.items()}, self._rewards[:T], self._dones[:T]

    def run(self) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------- Primitives
    def act(self, action: np.ndarray) -> None:
        obs, reward, terminated, _, _ = self.env.step(action)
        self._actions.append(np.asarray(action, dtype=np.float32))
        for k, v in obs.items():
            self._obs[k].append(v)
        self._rewards.append(reward)
        self._dones.append(terminated)
        if terminated or len(self._actions) >= self.max_steps:
            raise EpisodeDone

    def action_towards(self, target: np.ndarray, yaw: float | None, grip: float, max_step: float | None = None) -> np.ndarray:
        env = self.env
        error = np.asarray(target, dtype=np.float64) - env.tcp_position
        limit = env.max_tcp_step if max_step is None else min(max_step, env.max_tcp_step)
        dist = np.linalg.norm(error)
        speed = min(limit, np.sqrt(2.0 * self.accel_step * dist), dist)  # brake to stop at the target
        desired = error * (speed / dist) if dist > 1e-9 else np.zeros(3)
        change = desired - self._prev_delta
        change_norm = np.linalg.norm(change)
        if change_norm > self.accel_step:
            change *= self.accel_step / change_norm
        delta = self._prev_delta + change
        self._prev_delta = delta
        action = np.zeros(7)
        action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
        if yaw is not None:
            dyaw = (yaw - env.tcp_yaw + np.pi) % (2.0 * np.pi) - np.pi
            action[5] = np.clip(dyaw / env.yaw_scale, -1.0, 1.0)
        action[6] = grip
        return action

    def servo(self, target: Target, yaw: float | None, grip: float, tol: float = 0.002, max_steps: int = 80,
              max_step: float | None = None, until: Callable[[], bool] | None = None) -> bool:
        """Drive the TCP to a (possibly moving) target; returns True once within tol (and yaw within 2 deg)."""
        for _ in range(max_steps):
            goal = target() if callable(target) else target
            yaw_ok = yaw is None or abs((yaw - self.env.tcp_yaw + np.pi) % (2.0 * np.pi) - np.pi) < np.radians(2.0)
            if (np.linalg.norm(goal - self.env.tcp_position) < tol and yaw_ok) or (until is not None and until()):
                return True
            self.act(self.action_towards(goal, yaw, grip, max_step))
        return False

    def hold(self, grip: float, steps: int, yaw: float | None = None) -> None:
        target = self.env.tcp_position.copy()
        for _ in range(steps):
            self.act(self.action_towards(target, yaw, grip))
