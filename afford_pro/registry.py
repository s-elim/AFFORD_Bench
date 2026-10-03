"""Master Task Registry for AFFORD-PRO-150 Benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type
import gymnasium as gym

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.oracles.base_oracle import BaseOracle

# Industrial tasks
from afford_pro.envs.industrial.shaft_assembly import FrankaShaftAssemblyEnv
from afford_pro.oracles.industrial_oracle import ShaftAssemblyOracle
from afford_pro.envs.industrial.gear_mesh import FrankaGearMeshEnv
from afford_pro.oracles.gear_mesh_oracle import GearMeshOracle
from afford_pro.envs.industrial.pin_insertion import FrankaPinInsertionEnv
from afford_pro.oracles.pin_insertion_oracle import PinInsertionOracle

# Kitchen tasks
from afford_pro.envs.kitchen.patty_stack import FrankaPattyStackEnv
from afford_pro.oracles.patty_stack_oracle import PattyStackOracle
from afford_pro.envs.kitchen.cheese_place import FrankaCheesePlaceEnv
from afford_pro.oracles.cheese_place_oracle import CheesePlaceOracle
from afford_pro.envs.kitchen.coffee_mug import FrankaCoffeeMugEnv
from afford_pro.oracles.coffee_mug_oracle import CoffeeMugOracle

# Tabletop sorting tasks
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
    env_class: Type[BaseFrankaEnv]
    oracle_class: Type[BaseOracle]
    instruction: str
    description: str


_TASK_REGISTRY: dict[str, TaskEntry] = {
    # Domain 1: Industrial Precision Assembly (001 - 030)
    "industrial_shaft_assembly": TaskEntry(
        task_id="industrial_shaft_assembly",
        domain="Industrial & Manufacturing",
        task_number=1,
        env_class=FrankaShaftAssemblyEnv,
        oracle_class=ShaftAssemblyOracle,
        instruction="Pick the machined transmission shaft from the inbound tray and seat it into the central assembly jig bore.",
        description="High-precision insertion of cylindrical transmission shaft into cast-iron jig bore with tight tolerances.",
    ),
    "industrial_gear_mesh": TaskEntry(
        task_id="industrial_gear_mesh",
        domain="Industrial & Manufacturing",
        task_number=2,
        env_class=FrankaGearMeshEnv,
        oracle_class=GearMeshOracle,
        instruction="Pick the bronze spur gear from the feeder spindle and mesh it onto the transmission shaft in the jig.",
        description="Pick bronze spur gear from component rack and align onto the seated vertical shaft.",
    ),
    "industrial_pin_insertion": TaskEntry(
        task_id="industrial_pin_insertion",
        domain="Industrial & Manufacturing",
        task_number=3,
        env_class=FrankaPinInsertionEnv,
        oracle_class=PinInsertionOracle,
        instruction="Pick the hardened steel dowel pin from the feeder cup and insert it into the locating hole on the assembly jig.",
        description="Pick small dowel pin from feeder cup and press into fixture precision guide hole.",
    ),

    # Domain 2: Culinary & Commercial Kitchen (031 - 060)
    "kitchen_patty_stack": TaskEntry(
        task_id="kitchen_patty_stack",
        domain="Culinary & Kitchen Operations",
        task_number=31,
        env_class=FrankaPattyStackEnv,
        oracle_class=PattyStackOracle,
        instruction="Pick the grilled beef burger patty from the cast-iron grill and place it onto the bottom bun on the assembly board.",
        description="Transfer cooked patty from hot searing grill station onto burger base bread.",
    ),
    "kitchen_cheese_place": TaskEntry(
        task_id="kitchen_cheese_place",
        domain="Culinary & Kitchen Operations",
        task_number=32,
        env_class=FrankaCheesePlaceEnv,
        oracle_class=CheesePlaceOracle,
        instruction="Pick the cheddar cheese slice from the chilled topping tray and place it onto the burger patty.",
        description="Pick deformable planar cheddar cheese slice and align horizontally onto warm burger patty.",
    ),
    "kitchen_mug_to_espresso": TaskEntry(
        task_id="kitchen_mug_to_espresso",
        domain="Culinary & Kitchen Operations",
        task_number=36,
        env_class=FrankaCoffeeMugEnv,
        oracle_class=CoffeeMugOracle,
        instruction="Pick the ceramic coffee mug from the storage bin and place it onto the espresso machine drip tray.",
        description="Retrieve ceramic cup from clean caddy and place on recessed drip tray under the machine spout.",
    ),

    # Domain 3: Tabletop & Domestic Manipulation (061 - 090)
    "tabletop_sort_red": TaskEntry(
        task_id="tabletop_sort_red",
        domain="Tabletop & Domestic",
        task_number=61,
        env_class=FrankaSortRedEnv,
        oracle_class=BlockSortingOracle,
        instruction="Pick the red cube block and place it into the red sorting bin.",
        description="Color and geometry classification pick and place into designated sorting container.",
    ),
    "tabletop_sort_green": TaskEntry(
        task_id="tabletop_sort_green",
        domain="Tabletop & Domestic",
        task_number=62,
        env_class=FrankaSortGreenEnv,
        oracle_class=BlockSortingOracle,
        instruction="Pick the green cylinder block and place it into the green sorting bin.",
        description="Pick cylindrical green token and deposit into matching green bin.",
    ),
    "tabletop_sort_blue": TaskEntry(
        task_id="tabletop_sort_blue",
        domain="Tabletop & Domestic",
        task_number=63,
        env_class=FrankaSortBlueEnv,
        oracle_class=BlockSortingOracle,
        instruction="Pick the blue cube block and place it into the blue sorting bin.",
        description="Pick blue cubic block and deposit into matching blue bin.",
    ),
}


def list_tasks() -> list[str]:
    """Return all registered task IDs."""
    return list(_TASK_REGISTRY.keys())


def get_task_entry(task_id: str) -> TaskEntry:
    if task_id not in _TASK_REGISTRY:
        raise KeyError(f"Task '{task_id}' not found in AFFORD-PRO-150 registry. Available: {list_tasks()}")
    return _TASK_REGISTRY[task_id]


def make(task_id: str, **kwargs) -> BaseFrankaEnv:
    """Instantiate an AFFORD-PRO-150 environment by task ID."""
    entry = get_task_entry(task_id)
    return entry.env_class(**kwargs)


def get_oracle(task_id: str, env: BaseFrankaEnv) -> BaseOracle:
    """Instantiate the default algorithmic oracle for a given task and environment."""
    entry = get_task_entry(task_id)
    return entry.oracle_class(env)
