"""Policy inputs shared by training (from HDF5 datasets) and evaluation (from a live environment).

Modalities:
  state   robot proprioception only (TCP pose, joint positions and velocities, gripper opening; 22-D). It does not
          contain the object, so under the L1 start scatter it is a blind ablation, not a baseline.
  lowdim  proprioception + pose of the evaluated body (position and w-first quaternion; 29-D), decoded from the
          recorded simulator state. This is privileged object state, as in robomimic's low-dim datasets.
  objects proprioception + poses of the bodies the task moves, in goal order, in OBJECT_SLOTS zero-padded slots
          (64-D): the privileged low-dim input for tasks that move up to six bodies (one slot for legacy tasks).
  visual  proprioception + the workcell RGB image subsampled to 3 x 80 x 80.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MODALITIES = ("state", "lowdim", "objects", "visual")
PROPRIO_KEYS = ("tcp_pose", "joint_positions", "joint_velocities", "gripper_width")
PROPRIO_DIM = 22
OBJECT_POSE_DIM = 7
OBJECT_SLOTS = 6
IMAGE_STRIDE = 8
IMAGE_SIZE = 80
STD_FLOOR = 1e-3


def input_dim(modality: str) -> int:
    extra = {"lowdim": OBJECT_POSE_DIM, "objects": OBJECT_POSE_DIM * OBJECT_SLOTS}
    return PROPRIO_DIM + extra.get(modality, 0)


def proprio(obs: dict[str, Any], t: int | slice | None = None) -> np.ndarray:
    """Concatenate proprioception from an observation dict (one step) or an HDF5 obs group (index t)."""
    parts = [np.asarray(obs[k] if t is None else obs[k][t], dtype=np.float32) for k in PROPRIO_KEYS]
    return np.concatenate(parts, axis=-1)


def object_pose_from_states(states: np.ndarray, qpos_offset: int, object_qpos_adr: int) -> np.ndarray:
    start = qpos_offset + object_qpos_adr
    return np.asarray(states[..., start:start + OBJECT_POSE_DIM], dtype=np.float32)


def object_poses_from_states(states: np.ndarray, qpos_offset: int, qpos_adrs: list[int]) -> np.ndarray:
    """(T, OBJECT_SLOTS * 7) poses of the given free bodies, zero-padded."""
    states = np.asarray(states)
    out = np.zeros((*states.shape[:-1], OBJECT_POSE_DIM * OBJECT_SLOTS), dtype=np.float32)
    for k, adr in enumerate(list(qpos_adrs)[:OBJECT_SLOTS]):
        out[..., k * OBJECT_POSE_DIM:(k + 1) * OBJECT_POSE_DIM] = object_pose_from_states(states, qpos_offset, adr)
    return out


def moved_body_qpos_adrs(env: Any) -> list[int]:
    """qpos addresses of the bodies a task moves: every goal body of a task, the evaluated body of a legacy task."""
    if hasattr(env, "task_spec") and hasattr(env, "body_qposadr"):
        return [env.body_qposadr[b] for b in env.task_spec.bodies_moved()]
    return [] if getattr(env, "target_qposadr", None) is None else [env.target_qposadr]


def object_poses_from_env(env: Any) -> np.ndarray:
    out = np.zeros(OBJECT_POSE_DIM * OBJECT_SLOTS, dtype=np.float32)
    for k, adr in enumerate(moved_body_qpos_adrs(env)[:OBJECT_SLOTS]):
        out[k * OBJECT_POSE_DIM:(k + 1) * OBJECT_POSE_DIM] = env.data.qpos[adr:adr + OBJECT_POSE_DIM]
    return out


def object_pose_from_env(env: Any) -> np.ndarray:
    adr = env.target_qposadr
    if adr is None:
        return np.zeros(OBJECT_POSE_DIM, dtype=np.float32)
    return np.asarray(env.data.qpos[adr:adr + OBJECT_POSE_DIM], dtype=np.float32)


def preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 workcell image -> (3, 80, 80) float32 in [0, 1]."""
    img = np.asarray(rgb)[::IMAGE_STRIDE, ::IMAGE_STRIDE, :3][:IMAGE_SIZE, :IMAGE_SIZE, :]
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def env_inputs(obs: dict[str, Any], env: Any, modality: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Low-dim vector and optional image for one live step, matching the training features."""
    vec = proprio(obs)
    if modality == "lowdim":
        vec = np.concatenate([vec, object_pose_from_env(env)])
    elif modality == "objects":
        vec = np.concatenate([vec, object_poses_from_env(env)])
    image = preprocess_image(obs["agentic_rgb"]) if modality == "visual" else None
    return vec, image


@dataclass
class Normalizer:
    """Per-dimension z-scoring fitted on the training split and stored in the checkpoint."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        x = np.asarray(x, dtype=np.float64)
        return cls(x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), STD_FLOOR).astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]] | None, dim: int) -> "Normalizer":
        if d is None:  # checkpoints from before normalization was stored: identity
            return cls(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))
