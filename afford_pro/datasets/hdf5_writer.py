"""Robomimic and LIBERO compatible HDF5 dataset writer.

Layout:
  data/
    attrs:
      env_args: JSON environment metadata ({"env_name", "type", "env_kwargs"}) read by robomimic.
      problem_info: JSON problem description ({"domain_name", "problem_name", "language_instruction"}) read by LIBERO.
      env_name, task_id, instruction, total, num_demos, model_file, language_instructions, prompt_variants.
    demo_i/
      actions: (T, 7) float32
      rewards: (T,) float32
      dones: (T,) uint8
      states: (T, S) float64 full physics simulator state before actions[t]
      obs/
        # Canonical AFFORD-Bench keys:
        agentic_rgb, wrist_rgb, joint_positions, joint_velocities, tcp_pose (7-D), gripper_width (1-D)
        # Canonical Robomimic / robosuite keys (aliased/derived):
        agentview_rgb, eye_in_hand_rgb, agentview_image, robot0_eye_in_hand_image,
        robot0_joint_pos, robot0_joint_vel, robot0_eef_pos (3-D), robot0_eef_quat (4-D, scalar-last),
        robot0_gripper_qpos (2-D)
        # Canonical LIBERO keys:
        joint_states (7-D), gripper_states (2-D), ee_pos (3-D), ee_ori (3-D axis-angle), ee_states (6-D)
      next_obs/
        All corresponding next-step observations for offline RL (BCQ, CQL, TD3-BC) and robomimic SequenceDataset.
      attrs:
        num_samples: T
        seed: reset seed used for the demonstration
        model_file: MuJoCo model XML string (used by robomimic playback_dataset and LIBERO reset_from_xml_string)
        init_state: initial simulator state vector states[0] (used by LIBERO/robosuite state replay)
        language_instruction: task language instruction
  mask/
    train: demo keys for training split
    valid: demo keys for validation split
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np

from afford_pro.task_definitions import get_augmented_prompts

ROBOMIMIC_GYM_TYPE = 2  # robomimic EnvType.GYM_TYPE (robomimic 0.2.0, envs/env_base.py)

# Direct 1-to-1 key aliases
ROBOMIMIC_OBS_ALIASES = {
    "agentic_rgb": "agentview_rgb",
    "agentview_rgb": "agentview_image",
    "wrist_rgb": "eye_in_hand_rgb",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "joint_positions": "joint_states",
    "joint_states": "robot0_joint_pos",
    "joint_velocities": "robot0_joint_vel",
    "agentic_depth": "agentview_depth",
    "wrist_depth": "eye_in_hand_depth",
}


def _quat_wxyz_to_axisangle(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert w-first MuJoCo quaternions (..., 4) [qw, qx, qy, qz] to 3-D axis-angle vectors."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    small = s < 1e-7
    scale = np.zeros_like(angle)
    scale[~small] = angle[~small] / s[~small]
    return (v * scale[..., np.newaxis]).astype(np.float32)


def make_env_args(task_id: str, **env_kwargs: Any) -> str:
    """robomimic environment metadata for an AFFORD-Bench task: `afford_pro.make(env_name, **env_kwargs)` rebuilds
    the environment (robomimic itself cannot construct it; the entry lets its loaders read the dataset)."""
    return json.dumps({
        "env_name": task_id,
        "type": ROBOMIMIC_GYM_TYPE,
        "env_kwargs": {"factory": "afford_pro.make", **env_kwargs},
        "env_version": "0.1.0",
    })


class RobomimicHDF5Writer:
    """Creates robomimic- and LIBERO-compatible HDF5 demonstration datasets."""

    def __init__(
        self,
        file_path: str | Path,
        env_name: str,
        task_id: str,
        instruction: str,
        env_args: str | None = None,
        model_file: str | None = None,
        language_instructions: list[str] | None = None,
        prompt_variants: dict[str, Any] | None = None,
    ):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_name = env_name
        self.task_id = task_id
        self.instruction = instruction
        self.model_file = str(model_file) if model_file is not None else None
        self.demo_count = 0
        self.total_samples = 0

        # Open file and initialize groups
        self.h5_file = h5py.File(str(self.file_path), "w")
        self.data_grp = self.h5_file.create_group("data")
        self.mask_grp = self.h5_file.create_group("mask")

        # Root attributes (Robomimic & LIBERO canonical attributes)
        self.data_grp.attrs["env_name"] = env_name
        self.data_grp.attrs["env"] = env_name
        self.data_grp.attrs["task_id"] = task_id
        self.data_grp.attrs["instruction"] = instruction
        env_args_str = env_args if env_args is not None else make_env_args(task_id)
        self.data_grp.attrs["env_args"] = env_args_str
        self.data_grp.attrs["env_info"] = env_args_str
        self.data_grp.attrs["tag"] = "libero-v1"
        self.data_grp.attrs["repository_version"] = "0.1.0"
        self.data_grp.attrs["macros_image_convention"] = "opencv"
        self.data_grp.attrs["problem_info"] = json.dumps({
            "domain_name": env_name,
            "problem_name": task_id,
            "language_instruction": instruction,
        })
        if self.model_file is not None:
            self.data_grp.attrs["model_file"] = self.model_file

        # Multi-tier language prompt variants (for LIBERO-PRO/X paraphrase evaluation)
        if prompt_variants is None or language_instructions is None:
            aug = get_augmented_prompts(task_id)
            if aug:
                if prompt_variants is None:
                    prompt_variants = aug.get("prompts", {})
                if language_instructions is None:
                    variants = [instruction]
                    prompts = aug.get("prompts", {})
                    for tier in ("colloquial_terse", "geometric_spatial", "affordance_physical"):
                        t_text = prompts.get(tier)
                        if t_text and isinstance(t_text, str) and t_text not in variants:
                            variants.append(t_text)
                    language_instructions = variants

        if language_instructions is None:
            language_instructions = [instruction]
        self.language_instructions = language_instructions
        self.data_grp.attrs["language_instructions"] = json.dumps(language_instructions)
        if prompt_variants:
            self.data_grp.attrs["prompt_variants"] = json.dumps(prompt_variants)

    def _populate_obs_group(self, group: h5py.Group, obs_dict: dict[str, np.ndarray]) -> None:
        """Write observations and establish all canonical robosuite and LIBERO keys."""
        # 1. Write base observations
        for k, arr in obs_dict.items():
            if k not in group:
                chunks = (1, *arr.shape[1:]) if arr.ndim >= 3 else True
                group.create_dataset(k, data=arr, compression="gzip", chunks=chunks)

        # 2. Derive proprioception keys if tcp_pose is present
        if "tcp_pose" in group:
            tcp_arr = group["tcp_pose"][()]  # (T, 7) [x, y, z, qw, qx, qy, qz]
            ee_pos = tcp_arr[:, :3].astype(np.float32)
            # MuJoCo w-first to 3D axis-angle
            ee_ori = _quat_wxyz_to_axisangle(tcp_arr[:, 3:7])
            # LIBERO standard 6-D [ee_pos, ee_ori]
            ee_states = np.concatenate([ee_pos, ee_ori], axis=-1)
            # robosuite scalar-last quaternion [qx, qy, qz, qw]
            robot0_eef_quat = np.concatenate([tcp_arr[:, 4:7], tcp_arr[:, 3:4]], axis=-1).astype(np.float32)

            for k, v in [
                ("ee_pos", ee_pos),
                ("robot0_eef_pos", ee_pos),
                ("ee_ori", ee_ori),
                ("ee_states", ee_states),
                ("robot0_eef_quat", robot0_eef_quat),
            ]:
                if k not in group:
                    group.create_dataset(k, data=v, compression="gzip")

        # 3. Derive gripper keys if gripper_width is present
        if "gripper_width" in group:
            gw = group["gripper_width"][()]
            T = len(gw)
            if gw.ndim == 1:
                gw_flat = gw.reshape(T, 1).astype(np.float32)
                gripper_2d = np.repeat(gw_flat, 2, axis=-1)
            elif gw.ndim == 2 and gw.shape[1] == 1:
                gripper_2d = np.repeat(gw.astype(np.float32), 2, axis=-1)
            else:
                gripper_2d = gw.astype(np.float32)
            for k, v in [("gripper_states", gripper_2d), ("robot0_gripper_qpos", gripper_2d)]:
                if k not in group:
                    group.create_dataset(k, data=v, compression="gzip")

        # 4. Create standard aliases via hard links
        for src_key, alias_key in ROBOMIMIC_OBS_ALIASES.items():
            if src_key in group and alias_key not in group:
                group[alias_key] = group[src_key]
            elif alias_key in group and src_key not in group:
                group[src_key] = group[alias_key]

    def add_demonstration(
        self,
        actions: list[np.ndarray] | np.ndarray,
        observations: dict[str, list[np.ndarray] | np.ndarray],
        rewards: list[float] | np.ndarray | None = None,
        dones: list[bool] | np.ndarray | None = None,
        states: np.ndarray | None = None,
        seed: int | None = None,
        language_instruction: str | None = None,
        next_observations: dict[str, list[np.ndarray] | np.ndarray] | None = None,
    ) -> str:
        """Add a complete recorded episode to the dataset."""
        demo_key = f"demo_{self.demo_count}"
        ep_grp = self.data_grp.create_group(demo_key)

        act_arr = np.asarray(actions, dtype=np.float32)
        T = len(act_arr)
        ep_grp.create_dataset("actions", data=act_arr, compression="gzip")

        # Process observations
        obs_arrays = {k: np.asarray(v) for k, v in observations.items()}
        obs_grp = ep_grp.create_group("obs")
        self._populate_obs_group(obs_grp, obs_arrays)

        # Process next_observations for offline RL (robomimic SequenceDataset, BCQ, CQL, TD3-BC)
        next_obs_grp = ep_grp.create_group("next_obs")
        if next_observations is not None:
            next_obs_arrays = {k: np.asarray(v) for k, v in next_observations.items()}
        else:
            # Shifted observation sequence: next_obs[t] = obs[t+1] for t < T-1, and obs[T-1] for t = T-1
            next_obs_arrays = {}
            for k, arr in obs_arrays.items():
                if len(arr) == T:
                    shifted = np.empty_like(arr)
                    shifted[:-1] = arr[1:]
                    shifted[-1] = arr[-1]
                    next_obs_arrays[k] = shifted
        if next_obs_arrays:
            self._populate_obs_group(next_obs_grp, next_obs_arrays)

        # Rewards & dones
        rew_arr = np.asarray(rewards, dtype=np.float32) if rewards is not None else np.zeros(T, dtype=np.float32)
        done_arr = np.asarray(dones, dtype=np.uint8) if dones is not None else np.zeros(T, dtype=np.uint8)

        ep_grp.create_dataset("rewards", data=rew_arr)
        ep_grp.create_dataset("dones", data=done_arr)

        # States & initial state
        if states is not None and len(states):
            states_arr = np.asarray(states, dtype=np.float64)
            assert len(states_arr) == T, (states_arr.shape, T)
            ep_grp.create_dataset("states", data=states_arr, compression="gzip")
            # Per-demo init state attribute (LIBERO / robosuite standard)
            ep_grp.attrs["init_state"] = states_arr[0]

        # Per-demo attributes
        ep_grp.attrs["num_samples"] = T
        if seed is not None:
            ep_grp.attrs["seed"] = int(seed)
        if self.model_file is not None:
            ep_grp.attrs["model_file"] = self.model_file

        if language_instruction is None:
            language_instruction = self.language_instructions[self.demo_count % len(self.language_instructions)]
        ep_grp.attrs["language_instruction"] = language_instruction
        ep_grp.attrs["instruction"] = language_instruction

        self.demo_count += 1
        self.total_samples += T
        return demo_key

    def close(self, train_ratio: float = 0.9):
        """Finalize dataset, write train/validation splits, and close file."""
        demo_keys = [f"demo_{i}" for i in range(self.demo_count)]
        n_train = int(len(demo_keys) * train_ratio)
        train_keys = np.array(demo_keys[:n_train], dtype="S")
        valid_keys = np.array(demo_keys[n_train:], dtype="S")

        self.mask_grp.create_dataset("train", data=train_keys)
        self.mask_grp.create_dataset("valid", data=valid_keys)

        self.data_grp.attrs["total"] = self.total_samples
        self.data_grp.attrs["num_demos"] = self.demo_count
        self.h5_file.close()
        print(f"[RobomimicHDF5Writer] Closed {self.file_path}: {self.demo_count} demos ({self.total_samples} timesteps)")


# Alias for general usage
HDF5DatasetWriter = RobomimicHDF5Writer
