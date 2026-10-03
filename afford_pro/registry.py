"""Master Task Registry for the Complete AFFORD-PRO-150 Benchmark Suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type
import gymnasium as gym

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.oracles.base_oracle import BaseOracle
from afford_pro.envs.parametric_env import ParametricFrankaEnv
from afford_pro.oracles.parametric_oracle import ParametricOracle
from afford_pro.task_definitions import ALL_150_TASKS, TaskDef

# Specialized custom environments & oracles
from afford_pro.envs.industrial.shaft_assembly import FrankaShaftAssemblyEnv
from afford_pro.oracles.industrial_oracle import ShaftAssemblyOracle
from afford_pro.envs.industrial.gear_mesh import FrankaGearMeshEnv
from afford_pro.oracles.gear_mesh_oracle import GearMeshOracle
from afford_pro.envs.industrial.pin_insertion import FrankaPinInsertionEnv
from afford_pro.oracles.pin_insertion_oracle import PinInsertionOracle
from afford_pro.envs.kitchen.patty_stack import FrankaPattyStackEnv
from afford_pro.oracles.patty_stack_oracle import PattyStackOracle
from afford_pro.envs.kitchen.cheese_place import FrankaCheesePlaceEnv
from afford_pro.oracles.cheese_place_oracle import CheesePlaceOracle
from afford_pro.envs.kitchen.coffee_mug import FrankaCoffeeMugEnv
from afford_pro.oracles.coffee_mug_oracle import CoffeeMugOracle
from afford_pro.envs.tabletop.block_sorting import (
    FrankaSortRedEnv,
    FrankaSortGreenEnv,
    FrankaSortBlueEnv,
)
from afford_pro.oracles.block_sorting_oracle import BlockSortingOracle


@dataclass
class TaskEntry:
    task_id: str
    domain: str
    task_number: int
    env_factory: Callable[..., BaseFrankaEnv]
    oracle_factory: Callable[[BaseFrankaEnv], BaseOracle]
    instruction: str
    description: str


_CUSTOM_MAP: dict[str, tuple[Type[BaseFrankaEnv], Type[BaseOracle]]] = {
    "industrial_shaft_assembly": (FrankaShaftAssemblyEnv, ShaftAssemblyOracle),
    "industrial_gear_mesh": (FrankaGearMeshEnv, GearMeshOracle),
    "industrial_pin_insertion": (FrankaPinInsertionEnv, PinInsertionOracle),
    "kitchen_patty_stack": (FrankaPattyStackEnv, PattyStackOracle),
    "kitchen_cheese_place": (FrankaCheesePlaceEnv, CheesePlaceOracle),
    "kitchen_mug_to_espresso": (FrankaCoffeeMugEnv, CoffeeMugOracle),
    "tabletop_sort_red": (FrankaSortRedEnv, BlockSortingOracle),
    "tabletop_sort_green": (FrankaSortGreenEnv, BlockSortingOracle),
    "tabletop_sort_blue": (FrankaSortBlueEnv, BlockSortingOracle),
}


def _make_parametric_env_factory(task_def: TaskDef) -> Callable[..., BaseFrankaEnv]:
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
            **kwargs,
        )
    return factory


def _make_parametric_oracle_factory() -> Callable[[BaseFrankaEnv], BaseOracle]:
    def factory(env: BaseFrankaEnv) -> BaseOracle:
        return ParametricOracle(env)
    return factory


_TASK_REGISTRY: dict[str, TaskEntry] = {}

# Populate all 150 tasks
for t in ALL_150_TASKS:
    if t.task_id in _CUSTOM_MAP:
        env_cls, oracle_cls = _CUSTOM_MAP[t.task_id]
        _TASK_REGISTRY[t.task_id] = TaskEntry(
            task_id=t.task_id,
            domain=t.domain,
            task_number=t.task_number,
            env_factory=env_cls,
            oracle_factory=oracle_cls,
            instruction=t.instruction,
            description=t.description,
        )
    else:
        _TASK_REGISTRY[t.task_id] = TaskEntry(
            task_id=t.task_id,
            domain=t.domain,
            task_number=t.task_number,
            env_factory=_make_parametric_env_factory(t),
            oracle_factory=_make_parametric_oracle_factory(),
            instruction=t.instruction,
            description=t.description,
        )


def list_tasks() -> list[str]:
    """Return all 150 registered task IDs."""
    return list(_TASK_REGISTRY.keys())


def get_task_entry(task_id: str) -> TaskEntry:
    if task_id not in _TASK_REGISTRY:
        raise KeyError(f"Task '{task_id}' not found in AFFORD-PRO-150 registry. Available: {len(_TASK_REGISTRY)} tasks.")
    return _TASK_REGISTRY[task_id]


def make(task_id: str, **kwargs) -> BaseFrankaEnv:
    """Instantiate an AFFORD-PRO-150 environment by task ID."""
    entry = get_task_entry(task_id)
    return entry.env_factory(**kwargs)


def get_oracle(task_id: str, env: BaseFrankaEnv) -> BaseOracle:
    """Instantiate the default algorithmic oracle for a given task and environment."""
    entry = get_task_entry(task_id)
    return entry.oracle_factory(env)
