"""Closed-loop rollouts with one success definition for every policy.

An episode succeeds if the environment reports success at some control step t <= horizon (the protocol's H).
Each episode is reset from its own seed (and optionally a fixed initial state); policies are re-seeded per episode,
so diffusion sampling and random actions are reproducible. Every episode yields one EpisodeRecord.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

F_MAX = 150.0  # N; force-violation threshold of the protocol (main.tex Sec. III-A)


@dataclass
class EpisodeRecord:
    task_id: str
    suite: str
    policy: str
    seed: int
    success: bool
    steps: int
    first_success_step: int | None
    peak_force_n: float
    force_violation: bool
    wall_s: float
    start_fallback: bool = False   # the start layout fell back to the region centers
    goal_fraction: float | None = None  # fraction of the task's goals reached within the horizon (None: one goal)


class RandomPolicy:
    name = "random"

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict, env: Any) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=7).astype(np.float32)


class NullPolicy:
    name = "null"

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: dict, env: Any) -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)


class CheckpointPolicy:
    """A trained BC or diffusion checkpoint; executes each predicted chunk open-loop for `replan` steps."""

    def __init__(self, path: str | Path, device: str = "cpu", replan: int | None = None, allow_legacy: bool = False):
        import torch

        from afford_pro.learning.features import Normalizer, input_dim
        from afford_pro.learning.policies import BCPolicy, DiffusionPolicy

        self.torch = torch
        # Only checkpoints produced by scripts/train_policy.py are loaded (full pickle, not weights-only).
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        legacy = [k for k in ("normalizer", "modality", "policy", "input_dim") if k not in ckpt]
        if legacy and not allow_legacy:
            raise ValueError(f"{path}: checkpoint lacks {legacy}; it predates normalized training, and evaluating it "
                             "with defaults would feed it inputs it was not trained on (pass allow_legacy=True to force)")
        self.modality = ckpt.get("modality", "state")
        self.kind = ckpt.get("policy", "diffusion")
        self.chunk = int(ckpt.get("chunk_size", 16))
        dim = int(ckpt.get("input_dim", input_dim(self.modality)))
        if dim != input_dim(self.modality):
            raise ValueError(f"{path}: input_dim {dim} does not match modality {self.modality!r}")
        cls = DiffusionPolicy if self.kind == "diffusion" else BCPolicy
        self.model = cls(action_dim=7, chunk_size=self.chunk, proprio_dim=dim, modality=self.modality).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.normalizer = Normalizer.from_dict(ckpt.get("normalizer"), dim)
        if self.normalizer.mean.shape != (dim,):
            raise ValueError(f"{path}: normalizer has {self.normalizer.mean.shape[0]} dims, model expects {dim}")
        self.device = device
        self.replan = replan or self.chunk
        self.name = f"{self.kind}_{self.modality}:{Path(path).name}"

    def reset(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        self.buffer: list[np.ndarray] = []

    def act(self, obs: dict, env: Any) -> np.ndarray:
        from afford_pro.learning.features import env_inputs

        if not self.buffer:
            vec, image = env_inputs(obs, env, self.modality)
            x = self.torch.from_numpy(self.normalizer(vec)).unsqueeze(0).to(self.device)
            img = None if image is None else self.torch.from_numpy(image).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                out = self.model.sample_actions(x, img) if self.kind == "diffusion" else self.model(x, img)
            self.buffer = list(np.clip(out.squeeze(0).cpu().numpy(), -1.0, 1.0)[: self.replan])
        return self.buffer.pop(0)


def goal_fraction(env: Any, horizon: int) -> float | None:
    """Fraction of the task's goals reached within `horizon` (goal-based tasks); None for single-goal legacy tasks."""
    spec, reached = getattr(env, "task_spec", None), getattr(env, "first_true", None)
    if spec is None or reached is None:
        return None
    return sum(step <= horizon for step in reached.values()) / len(spec.goals)


def run_policy_episode(env: Any, policy: Any, task_id: str, suite: str, seed: int, horizon: int,
                       init_state: np.ndarray | None = None) -> EpisodeRecord:
    t0 = time.monotonic()
    obs, _ = env.reset(seed=seed)
    if init_state is not None:
        env.set_sim_state(init_state)
        obs = env.get_observation()
    policy.reset(seed)
    peak, first = 0.0, None
    for t in range(horizon):
        obs, _, terminated, truncated, info = env.step(policy.act(obs, env))
        peak = max(peak, float(info.get("robot_scene_force", 0.0)))
        if terminated:
            first = t + 1
            break
        if truncated:
            break
    steps = first if first is not None else t + 1
    return EpisodeRecord(task_id, suite, policy.name, seed, first is not None, steps, first, peak, peak > F_MAX,
                         time.monotonic() - t0, bool(getattr(env, "start_fallback", False)), goal_fraction(env, horizon))


def run_oracle_episode(env: Any, oracle: Any, task_id: str, suite: str, seed: int, horizon: int) -> EpisodeRecord:
    """The privileged oracle runs up to its own budget; success counts only if reached within `horizon`."""
    t0 = time.monotonic()
    actions, _, _, _ = oracle.generate_episode(seed=seed)
    steps = len(actions)
    success = bool(env.check_success()) and steps <= horizon
    return EpisodeRecord(task_id, suite, "oracle", seed, success, steps, steps if success else None, float("nan"),
                         False, time.monotonic() - t0, bool(getattr(env, "start_fallback", False)),
                         goal_fraction(env, horizon))
