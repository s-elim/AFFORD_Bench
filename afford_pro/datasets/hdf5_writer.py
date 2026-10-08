"""Robomimic-format HDF5 dataset writer.

Layout: data/demo_i/{actions, obs/<key>, rewards, dones, states}, attrs on `data` (env_args, env_name, task_id,
instruction, total), num_samples and seed on each demo, and mask/{train, valid}. `env_args` is the JSON environment
metadata that robomimic reads at startup ({"env_name", "type", "env_kwargs"}); `states` holds the simulator state
before each action (BaseFrankaEnv.get_sim_state), so demos can be re-rendered after reset(seed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np


from afford_pro.task_definitions import get_augmented_prompts

ROBOMIMIC_GYM_TYPE = 2  # robomimic EnvType.GYM_TYPE (robomimic 0.2.0, envs/env_base.py)

# Standard aliases between AFFORD-Bench keys and canonical Robomimic / LIBERO loader keys
ROBOMIMIC_OBS_ALIASES = {
    "agentic_rgb": "agentview_rgb",
    "wrist_rgb": "eye_in_hand_rgb",
    "tcp_pose": "ee_states",
    "joint_positions": "joint_states",
    "agentic_depth": "agentview_depth",
    "wrist_depth": "eye_in_hand_depth",
}


def make_env_args(task_id: str, **env_kwargs: Any) -> str:
    """robomimic environment metadata for an AFFORD-Bench task: `afford_pro.make(env_name, **env_kwargs)` rebuilds
    the environment (robomimic itself cannot construct it; the entry lets its loaders read the dataset)."""
    return json.dumps({"env_name": task_id, "type": ROBOMIMIC_GYM_TYPE,
                       "env_kwargs": {"factory": "afford_pro.make", **env_kwargs}})


class RobomimicHDF5Writer:
    """Creates robomimic-format HDF5 demonstration datasets."""

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
        self.demo_count = 0
        self.total_samples = 0

        # Open file and initialize groups
        self.h5_file = h5py.File(str(self.file_path), "w")
        self.data_grp = self.h5_file.create_group("data")
        self.mask_grp = self.h5_file.create_group("mask")

        # Root attributes
        self.data_grp.attrs["env_name"] = env_name
        self.data_grp.attrs["task_id"] = task_id
        self.data_grp.attrs["instruction"] = instruction
        self.data_grp.attrs["env_args"] = env_args if env_args is not None else make_env_args(task_id)
        if model_file is not None:
            self.data_grp.attrs["model_file"] = str(model_file)

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

    def add_demonstration(
        self,
        actions: list[np.ndarray] | np.ndarray,
        observations: dict[str, list[np.ndarray] | np.ndarray],
        rewards: list[float] | np.ndarray | None = None,
        dones: list[bool] | np.ndarray | None = None,
        states: np.ndarray | None = None,
        seed: int | None = None,
        language_instruction: str | None = None,
    ) -> str:
        """Add a complete recorded episode to the dataset."""
        demo_key = f"demo_{self.demo_count}"
        ep_grp = self.data_grp.create_group(demo_key)

        act_arr = np.asarray(actions, dtype=np.float32)
        T = len(act_arr)
        ep_grp.create_dataset("actions", data=act_arr, compression="gzip")

        # Observations
        obs_grp = ep_grp.create_group("obs")
        for k, v in observations.items():
            arr = np.asarray(v)
            # Use chunks for large image arrays
            chunks = (1, *arr.shape[1:]) if arr.ndim >= 3 else True
            obs_grp.create_dataset(k, data=arr, compression="gzip", chunks=chunks)

        # Robomimic / LIBERO observation key aliases (hard links to avoid duplicate storage)
        for src_key, alias_key in ROBOMIMIC_OBS_ALIASES.items():
            if src_key in obs_grp and alias_key not in obs_grp:
                obs_grp[alias_key] = obs_grp[src_key]
            elif alias_key in obs_grp and src_key not in obs_grp:
                obs_grp[src_key] = obs_grp[alias_key]

        # Rewards & dones
        rew_arr = np.asarray(rewards, dtype=np.float32) if rewards is not None else np.zeros(T, dtype=np.float32)
        done_arr = np.asarray(dones, dtype=np.uint8) if dones is not None else np.zeros(T, dtype=np.uint8)

        ep_grp.create_dataset("rewards", data=rew_arr)
        ep_grp.create_dataset("dones", data=done_arr)
        if states is not None and len(states):
            states = np.asarray(states, dtype=np.float64)
            assert len(states) == T, (states.shape, T)
            ep_grp.create_dataset("states", data=states, compression="gzip")
        ep_grp.attrs["num_samples"] = T
        if seed is not None:
            ep_grp.attrs["seed"] = int(seed)

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
        self.h5_file.close()
        print(f"[RobomimicHDF5Writer] Closed {self.file_path}: {self.demo_count} demos ({self.total_samples} timesteps)")


# Alias for general usage
HDF5DatasetWriter = RobomimicHDF5Writer

