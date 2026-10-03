"""Robomimic / LIBERO Compatible HDF5 Dataset Writer."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import h5py
import numpy as np


class RobomimicHDF5Writer:
    """Creates standard Robomimic / LIBERO HDF5 demonstration datasets."""

    def __init__(self, file_path: str | Path, env_name: str, task_id: str, instruction: str):
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

    def add_demonstration(
        self,
        actions: list[np.ndarray] | np.ndarray,
        observations: dict[str, list[np.ndarray] | np.ndarray],
        rewards: list[float] | np.ndarray | None = None,
        dones: list[bool] | np.ndarray | None = None,
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

        # Rewards & dones
        rew_arr = np.asarray(rewards, dtype=np.float32) if rewards is not None else np.zeros(T, dtype=np.float32)
        done_arr = np.asarray(dones, dtype=np.uint8) if dones is not None else np.zeros(T, dtype=np.uint8)

        ep_grp.create_dataset("rewards", data=rew_arr)
        ep_grp.create_dataset("dones", data=done_arr)
        ep_grp.attrs["num_samples"] = T

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

