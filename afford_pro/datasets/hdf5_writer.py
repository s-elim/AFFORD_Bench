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
        final_state: simulator state after the last action (the success state), when the recorder provides it
        language_instruction: task language instruction (the canonical instruction unless variants are rotated)
  mask/
    train: demo keys for training split
    valid: demo keys for validation split

The file is written to `<name>.partial` and renamed to its final name only by `close()`, so an interrupted run never
leaves a truncated file under the final name.
"""
from __future__ import annotations

import json
import os
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
    "joint_torques": "robot0_joint_torque",
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


def _shifted_next_obs(obs_arrays: dict[str, np.ndarray], T: int,
                      final_observation: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """next_obs[t] = obs[t+1] for t < T-1; next_obs[T-1] is the final observation when given, else obs[T-1]."""
    out = {}
    for k, arr in obs_arrays.items():
        if len(arr) != T:
            continue
        shifted = np.empty_like(arr)
        shifted[:-1] = arr[1:]
        last = None if final_observation is None else final_observation.get(k)
        shifted[-1] = arr[-1] if last is None else np.asarray(last, dtype=arr.dtype)
        out[k] = shifted
    return out


def _language_variants(task_id: str, instruction: str, language_instructions: list[str] | None,
                       prompt_variants: dict[str, Any] | None) -> tuple[list[str], dict[str, Any] | None]:
    """Instruction variants for the file: the canonical instruction first, then the task's terse, metric, and
    affordance tiers from configs/prompts_augmented.yaml, unless the caller supplies them."""
    if prompt_variants is None or language_instructions is None:
        prompts = (get_augmented_prompts(task_id) or {}).get("prompts", {})
        if prompt_variants is None and prompts:
            prompt_variants = prompts
        if language_instructions is None and prompts:
            language_instructions = [instruction]
            for tier in ("colloquial_terse", "geometric_spatial", "affordance_physical"):
                text = prompts.get(tier)
                if text and isinstance(text, str) and text not in language_instructions:
                    language_instructions.append(text)
    return (language_instructions or [instruction]), prompt_variants


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
        provenance: dict[str, Any] | None = None,
        atomic: bool = True,
        rotate_instruction_variants: bool = False,
    ):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process temporary name: two concurrent writers of the same dataset never share a file.
        self.write_path = (self.file_path.with_name(f"{self.file_path.name}.{os.getpid()}.partial") if atomic
                           else self.file_path)
        self.env_name = env_name
        self.task_id = task_id
        self.instruction = instruction
        self.model_file = str(model_file) if model_file is not None else None
        self.rotate_instruction_variants = rotate_instruction_variants
        self.demo_count = 0
        self.total_samples = 0

        # Open file and initialize groups
        self.h5_file = h5py.File(str(self.write_path), "w")
        self.data_grp = self.h5_file.create_group("data")
        self.mask_grp = self.h5_file.create_group("mask")
        self._write_root_attrs(env_args)

        # Multi-tier language prompt variants (for LIBERO-PRO/X paraphrase evaluation)
        self.language_instructions, prompt_variants = _language_variants(
            task_id, instruction, language_instructions, prompt_variants)
        self.data_grp.attrs["language_instructions"] = json.dumps(self.language_instructions)
        if prompt_variants:
            self.data_grp.attrs["prompt_variants"] = json.dumps(prompt_variants)
        if provenance is not None:
            self.data_grp.attrs["provenance"] = json.dumps(provenance, sort_keys=True, default=str)

    def _write_root_attrs(self, env_args: str | None) -> None:
        """Robomimic and LIBERO canonical root attributes."""
        attrs = self.data_grp.attrs
        attrs["env_name"] = self.env_name
        attrs["env"] = self.env_name
        attrs["task_id"] = self.task_id
        attrs["instruction"] = self.instruction
        env_args_str = env_args if env_args is not None else make_env_args(self.task_id)
        attrs["env_args"] = env_args_str
        attrs["env_info"] = env_args_str
        attrs["tag"] = "libero-v1"
        attrs["repository_version"] = "0.1.0"
        attrs["macros_image_convention"] = "opencv"
        attrs["problem_info"] = json.dumps({
            "domain_name": self.env_name,
            "problem_name": self.task_id,
            "language_instruction": self.instruction,
        })
        if self.model_file is not None:
            attrs["model_file"] = self.model_file

    def _populate_obs_group(self, group: h5py.Group, obs_dict: dict[str, np.ndarray]) -> None:
        """Write observations and establish all canonical robosuite and LIBERO keys."""
        # 1. Write base observations (skip alias targets whose source key is present so Step 4 links them)
        derived_or_aliased = set(ROBOMIMIC_OBS_ALIASES.values()) | {
            "ee_pos", "robot0_eef_pos", "ee_ori", "ee_states", "robot0_eef_quat", "ee_quat",
            "gripper_states", "robot0_gripper_qpos", "wrist_image",
        }
        for k, arr in obs_dict.items():
            if k in derived_or_aliased and (
                "agentic_rgb" in obs_dict or "wrist_rgb" in obs_dict or "tcp_pose" in obs_dict or "joint_torques" in obs_dict
            ):
                continue
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
        final_observation: dict[str, np.ndarray] | None = None,
        final_state: np.ndarray | None = None,
    ) -> str:
        """Add a complete recorded episode to the dataset.

        `final_observation` / `final_state` describe the scene after the last action. When given, next_obs[T-1]
        is that observation (otherwise it repeats obs[T-1], and the success state is not stored)."""
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
            next_obs_arrays = _shifted_next_obs(obs_arrays, T, final_observation)
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
            if len(states_arr) != T:
                raise ValueError(f"{len(states_arr)} states for {T} actions in {demo_key}")
            ep_grp.create_dataset("states", data=states_arr, compression="gzip")
            # Per-demo init state attribute (LIBERO / robosuite standard)
            ep_grp.attrs["init_state"] = states_arr[0]
        if final_state is not None:
            ep_grp.attrs["final_state"] = np.asarray(final_state, dtype=np.float64)
        self._write_episode_attrs(ep_grp, T, seed, language_instruction)

        self.demo_count += 1
        self.total_samples += T
        return demo_key

    def _write_episode_attrs(self, ep_grp: h5py.Group, T: int, seed: int | None, language_instruction: str | None) -> None:
        ep_grp.attrs["num_samples"] = T
        if seed is not None:
            ep_grp.attrs["seed"] = int(seed)
        if self.model_file is not None:
            ep_grp.attrs["model_file"] = self.model_file
        if language_instruction is None:
            # Variants with metric coordinates describe the nominal layout, not this episode's scattered start, so
            # they label a demo only when rotation is requested explicitly.
            language_instruction = (
                self.language_instructions[self.demo_count % len(self.language_instructions)]
                if self.rotate_instruction_variants else self.instruction
            )
        ep_grp.attrs["language_instruction"] = language_instruction
        ep_grp.attrs["instruction"] = language_instruction

    def close(self, train_ratio: float = 0.9, train_keys: list[str] | None = None,
              valid_keys: list[str] | None = None, extra_attrs: dict[str, Any] | None = None,
              rename: bool = True) -> Path:
        """Write train/validation masks and totals and close the file. With `rename` (default) the file is moved to its
        final name and that path is returned; without it the closed file stays at `write_path` (returned), so the
        caller can validate it before moving it into place."""
        demo_keys = [f"demo_{i}" for i in range(self.demo_count)]
        n_train = int(len(demo_keys) * train_ratio)
        train = demo_keys[:n_train] if train_keys is None else list(train_keys)
        valid = demo_keys[n_train:] if valid_keys is None else list(valid_keys)

        self.mask_grp.create_dataset("train", data=np.array(train, dtype="S"))
        self.mask_grp.create_dataset("valid", data=np.array(valid, dtype="S"))

        self.data_grp.attrs["total"] = self.total_samples
        self.data_grp.attrs["num_demos"] = self.demo_count
        for key, value in (extra_attrs or {}).items():
            self.data_grp.attrs[key] = value if isinstance(value, (str, int, float, np.ndarray)) else json.dumps(value, default=str)
        self.h5_file.close()
        if not rename:
            return self.write_path
        if self.write_path != self.file_path:
            os.replace(self.write_path, self.file_path)
        print(f"[RobomimicHDF5Writer] Closed {self.file_path}: {self.demo_count} demos ({self.total_samples} timesteps)")
        return self.file_path

    def abort(self) -> None:
        """Close and delete an unfinished file; the final path is never touched."""
        self.h5_file.close()
        if self.write_path != self.file_path and self.write_path.exists():
            self.write_path.unlink()


# Alias for general usage
HDF5DatasetWriter = RobomimicHDF5Writer
