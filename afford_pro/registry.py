"""Task registry: every AFFORD-Bench task is a ParametricFrankaEnv solved by the closed-loop ParametricOracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.envs.parametric_env import ParametricFrankaEnv
from afford_pro.oracles.base_oracle import BaseOracle
from afford_pro.oracles.parametric_oracle import ParametricOracle
from afford_pro.task_definitions import ALL_210_TASKS, TaskDef


@dataclass
class TaskEntry:
    task_id: str
    domain: str
    task_number: int
    env_factory: Callable[..., BaseFrankaEnv]
    oracle_factory: Callable[[BaseFrankaEnv], BaseOracle]
    instruction: str
    description: str


def _make_env_factory(task_def: TaskDef) -> Callable[..., BaseFrankaEnv]:
    def factory(**kwargs) -> BaseFrankaEnv:
        return ParametricFrankaEnv(
            scene_xml_name=task_def.scene_xml,
            object_name=task_def.object_name,
            joint_name=task_def.joint_name,
            nominal_pos=task_def.nominal_pos,
            target_pos=task_def.target_pos,
            instruction=task_def.instruction,
            task_type=task_def.task_type,
            tolerance_xy=task_def.tolerance_xy,
            tolerance_z=task_def.tolerance_z,
            tilt_tol_deg=task_def.tilt_tol_deg,
            min_lift=task_def.min_lift,
            init_tcp_pos=task_def.init_tcp_pos,
            scene_bodies=task_def.scene_bodies,
            distractors=task_def.distractors,
            goal_ref_body=task_def.goal_ref_body,
            **kwargs,
        )
    return factory


_TASK_REGISTRY: dict[str, TaskEntry] = {
    t.task_id: TaskEntry(
        task_id=t.task_id,
        domain=t.domain,
        task_number=t.task_number,
        env_factory=_make_env_factory(t),
        oracle_factory=ParametricOracle,
        instruction=t.instruction,
        description=t.description,
    )
    for t in ALL_210_TASKS
}


def list_tasks() -> list[str]:
    """Return all registered task IDs in task-number order."""
    return list(_TASK_REGISTRY.keys())


def get_task_entry(task_id: str) -> TaskEntry:
    if task_id not in _TASK_REGISTRY:
        raise KeyError(f"Task '{task_id}' not found in the AFFORD-Bench registry ({len(_TASK_REGISTRY)} tasks).")
    return _TASK_REGISTRY[task_id]


def make(task_id: str, **kwargs) -> BaseFrankaEnv:
    """Instantiate an AFFORD-Bench environment by task ID."""
    return get_task_entry(task_id).env_factory(**kwargs)


def get_oracle(task_id: str, env: BaseFrankaEnv) -> BaseOracle:
    """Instantiate the closed-loop oracle for a given task and environment."""
    return get_task_entry(task_id).oracle_factory(env)
