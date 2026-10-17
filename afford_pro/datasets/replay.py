"""Replay recorded demonstrations under the current code and physics.

A demo is consistent with the installed simulator when re-executing its actions from `states[0]` reproduces its
state trajectory and reaches success exactly at its last step. MuJoCo's recorded full-physics state omits the
solver warm start, so small numerical differences are expected; the report gives the largest deviation of the
evaluated body and of the robot joints rather than demanding bit equality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class ReplayResult:
    demo: str
    T: int
    success_at_end: bool
    first_success_step: int | None
    max_object_dev_m: float
    max_joint_dev_rad: float

    @property
    def consistent(self) -> bool:
        return self.success_at_end and self.first_success_step == self.T


def replay_demo(env: Any, states: np.ndarray, actions: np.ndarray, seed: int | None, demo: str,
                qpos_offset: int, object_qpos_adr: int | None, arm_qpos_adr: list[int]) -> ReplayResult:
    env.reset(seed=seed)            # re-applies the episode's model perturbations (none in the nominal suite)
    env.set_sim_state(states[0])
    first_success = None
    obj_dev, joint_dev = 0.0, 0.0
    arm_idx = [qpos_offset + a for a in arm_qpos_adr]
    for t, action in enumerate(actions):
        _, _, terminated, _, _ = env.step(action)
        if terminated and first_success is None:
            first_success = t + 1
        if t + 1 < len(states):
            now = env.get_sim_state()
            joint_dev = max(joint_dev, float(np.abs(now[arm_idx] - states[t + 1][arm_idx]).max()))
            if object_qpos_adr is not None:
                i = qpos_offset + object_qpos_adr
                obj_dev = max(obj_dev, float(np.linalg.norm(now[i:i + 3] - states[t + 1][i:i + 3])))
    return ReplayResult(demo, len(actions), bool(env.check_success()), first_success, obj_dev, joint_dev)


def replay_file(path: Path, demos: list[str], qpos_offset: int, object_qpos_adr: int | None,
                arm_qpos_adr: list[int]) -> list[dict[str, Any]]:
    """Replay the listed demos of one file headlessly; returns one dict per demo."""
    import afford_pro
    from afford_pro.datasets.validation import parse_dataset_name
    from afford_pro.envs.headless import headless

    task_id, mode = parse_dataset_name(Path(path))
    results = []
    with headless(), h5py.File(path, "r") as f:
        env = afford_pro.make(task_id, perturbation_mode=mode)
        try:
            for demo in demos:
                g = f["data"][demo]
                seed = g.attrs.get("seed")
                res = replay_demo(env, g["states"][()], g["actions"][()], None if seed is None else int(seed),
                                  demo, qpos_offset, object_qpos_adr, arm_qpos_adr)
                results.append({**asdict(res), "consistent": res.consistent})
        finally:
            env.close()
    return results
