"""Task registry: every AFFORD-Bench task is a ParametricFrankaEnv solved by the closed-loop ParametricOracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.envs.frontier_env import FrontierComplexEnv, TwistLockEnv
from afford_pro.envs.parametric_env import ParametricFrankaEnv
from afford_pro.oracles.base_oracle import BaseOracle
from afford_pro.oracles.frontier_oracle import FrontierOracle
from afford_pro.oracles.twist_lock_oracle import TwistLockOracle
from afford_pro.oracles.parametric_oracle import ParametricOracle
from afford_pro.task_definitions import ALL_210_TASKS, DOMAIN_8_FRONTIER_TASKS, TaskDef


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
        is_frontier = (
            task_def.domain == "Frontier Complex Manipulation"
            or task_def.task_type in ("reactive_perturbation", "push_and_grasp")
            or getattr(task_def, "perturbation_step", None) is not None
            or getattr(task_def, "min_push_distance", 0.0) > 0.0
        )
        if task_def.task_type == "twist_lock":
            return TwistLockEnv(
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
                mechanism=task_def.mechanism,
                **kwargs,
            )
        if is_frontier:
            return FrontierComplexEnv(
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
                perturbation_step=getattr(task_def, "perturbation_step", None),
                perturbation_offset=getattr(task_def, "perturbation_offset", None),
                second_perturbation_step=getattr(task_def, "second_perturbation_step", None),
                second_perturbation_offset=getattr(task_def, "second_perturbation_offset", None),
                goal_shift_step=getattr(task_def, "goal_shift_step", None),
                goal_shift_offset=getattr(task_def, "goal_shift_offset", None),
                min_push_distance=getattr(task_def, "min_push_distance", 0.0),
                **kwargs,
            )
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

# Register Domain 8 Frontier Tasks
for t in DOMAIN_8_FRONTIER_TASKS:
    entry = TaskEntry(
        task_id=t.task_id,
        domain=t.domain,
        task_number=t.task_number,
        env_factory=_make_env_factory(t),
        oracle_factory=TwistLockOracle if t.task_type == "twist_lock" else FrontierOracle,
        instruction=t.instruction,
        description=t.description,
    )
    _TASK_REGISTRY[t.task_id] = entry
    # Register aliases
    short_alias = t.task_id.replace("frontier_", "")
    if short_alias not in _TASK_REGISTRY:
        _TASK_REGISTRY[short_alias] = entry
    if f"{short_alias}_perturbation" not in _TASK_REGISTRY:
        _TASK_REGISTRY[f"{short_alias}_perturbation"] = entry

# Backward-compatibility alias for previous push-and-grasp task ID
if "frontier_overhang_and_grasp_plate" in _TASK_REGISTRY:
    _TASK_REGISTRY["frontier_nonprehensile_push_and_grasp"] = _TASK_REGISTRY["frontier_overhang_and_grasp_plate"]
    _TASK_REGISTRY["nonprehensile_push_and_grasp"] = _TASK_REGISTRY["frontier_overhang_and_grasp_plate"]


def list_tasks() -> list[str]:
    """Return all 210 core benchmark task IDs in task-number order."""
    return [t.task_id for t in ALL_210_TASKS]


def list_frontier_tasks() -> list[str]:
    """Return Domain 8 frontier complex task IDs."""
    return [t.task_id for t in DOMAIN_8_FRONTIER_TASKS]


def list_all_tasks(include_frontier: bool = False) -> list[str]:
    """Return task IDs, optionally including Domain 8 frontier tasks."""
    if include_frontier:
        return [t.task_id for t in ALL_210_TASKS] + [t.task_id for t in DOMAIN_8_FRONTIER_TASKS]
    return [t.task_id for t in ALL_210_TASKS]


def get_task_entry(task_id: str) -> TaskEntry:
    if task_id not in _TASK_REGISTRY:
        raise KeyError(f"Task '{task_id}' not found in the AFFORD-Bench registry ({len(_TASK_REGISTRY)} tasks).")
    return _TASK_REGISTRY[task_id]


def make(task_id: str, **kwargs) -> BaseFrankaEnv:
    """Instantiate an AFFORD-Bench environment by task ID."""
    kwargs = dict(kwargs)
    # Framework arguments to absorb/ignore (Robomimic, Robosuite, LIBERO, Gymnasium)
    for ignored_key in (
        "factory", "env_name", "lang", "ignore_done", "use_object_obs",
        "use_camera_obs", "camera_names", "camera_height", "camera_width",
        "camera_heights", "camera_widths", "reward_shaping", "control_freq",
        "gripper_visualization", "bddl_file_name", "bddl_file", "problem_name",
    ):
        kwargs.pop(ignored_key, None)

    # Translate depth flag
    if kwargs.pop("use_depth_obs", False) or kwargs.pop("camera_depths", False):
        kwargs["render_depth"] = True

    # Translate render flags if render_mode not explicitly provided
    if "render_mode" not in kwargs:
        if kwargs.pop("render", False) or kwargs.pop("has_renderer", False):
            kwargs["render_mode"] = "human"
        elif kwargs.pop("render_offscreen", False) or kwargs.pop("has_offscreen_renderer", False):
            kwargs["render_mode"] = "rgb_array"
    else:
        kwargs.pop("render", None)
        kwargs.pop("has_renderer", None)
        kwargs.pop("render_offscreen", None)
        kwargs.pop("has_offscreen_renderer", None)

    return get_task_entry(task_id).env_factory(**kwargs)


def get_oracle(task_id: str, env: BaseFrankaEnv) -> BaseOracle:
    """Instantiate the closed-loop oracle for a given task and environment."""
    return get_task_entry(task_id).oracle_factory(env)


from pathlib import Path
import numpy as np


def get_initial_states(task_id: str, init_states_dir: Path | str | None = None) -> np.ndarray | None:
    """Load pre-generated fixed initial simulator states for task_id, if available."""
    if init_states_dir is None:
        init_states_dir = Path(__file__).resolve().parent.parent / "data" / "init_states"
    path = Path(init_states_dir) / f"{task_id}_init_states.npz"
    if path.exists():
        try:
            data = np.load(str(path))
            return data["states"]
        except Exception:
            return None
    return None


# ----------------------------------------------------------------- Gym & Gymnasium Registration
# Registers all benchmark environments so robomimic EnvGym and standard RL frameworks can instantiate
# environments via gym.make(task_id) or gymnasium.make(task_id).
try:
    import gymnasium as _gymnasium
    _gym_reg = getattr(_gymnasium.envs, "registry", {})
    for _tid in list(_TASK_REGISTRY.keys()):
        try:
            _ns_id = f"afford_pro/{_tid}-v0"
            if _ns_id not in _gym_reg:
                _gymnasium.register(
                    id=_ns_id,
                    entry_point="afford_pro.registry:make",
                    kwargs={"task_id": _tid},
                    disable_env_checker=True,
                )
            if _tid not in _gym_reg:
                _gymnasium.register(
                    id=_tid,
                    entry_point="afford_pro.registry:make",
                    kwargs={"task_id": _tid},
                    disable_env_checker=True,
                )
        except Exception:
            pass
except Exception:
    pass

try:
    import gym as _gym
    _legacy_reg = getattr(_gym.envs, "registry", {})
    for _tid in list(_TASK_REGISTRY.keys()):
        try:
            _ns_id = f"afford_pro/{_tid}-v0"
            if _ns_id not in _legacy_reg:
                _gym.register(
                    id=_ns_id,
                    entry_point="afford_pro.registry:make",
                    kwargs={"task_id": _tid},
                )
            if _tid not in _legacy_reg:
                _gym.register(
                    id=_tid,
                    entry_point="afford_pro.registry:make",
                    kwargs={"task_id": _tid},
                )
        except Exception:
            pass
except Exception:
    pass


