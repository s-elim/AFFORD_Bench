"""Builder shared by the task modules: each task replaces the legacy task with the same number, keeping its domain
and recording the legacy id and instruction it replaces."""
from __future__ import annotations

from typing import Mapping

from afford_pro.task_definitions import ALL_210_TASKS
from afford_pro.bench.spec import Goal, Start, TaskSpec

LEGACY_BY_NUMBER = {t.task_number: t for t in ALL_210_TASKS}

# Reasons shared by many redefinitions (the per-task `change` strings refer to them).
NO_SUCH_OBJECT = "the legacy task named an object the scene did not simulate"
CHAIN_ENDED_EARLY = "the legacy task ended when the first body reached its goal, before the rest of the instruction"
NOT_SIMULABLE = "the legacy task named steps rigid-body simulation cannot represent (liquids, cutting, cooking, deformables)"
FIXED_POINT = "the legacy task scored hovering at a fixed point; replaced by a manipulation of a body whose pose varies"
# The bearing ring's 21.5 mm bore rides on the shaft's 16 mm core below the collar, so it can rest up to 5.5 mm off the
# shaft axis: "over the shaft" is checked to 6 mm, which still requires the ring to encircle the shaft.
BEARING_TOL = 0.006


def task(number: int, task_id: str, scene: str, instruction: str, *goals: Goal,
         starts: Mapping[str, Start] | None = None, sliders: Mapping[str, float] | None = None,
         order: tuple[tuple[int, int], ...] = (), change: str = "", max_steps: int | None = None) -> TaskSpec:
    legacy = LEGACY_BY_NUMBER[number]
    return TaskSpec(number=number, task_id=task_id, domain=legacy.domain, scene=scene, instruction=instruction,
                    goals=tuple(goals), starts=dict(starts or {}), sliders=dict(sliders or {}), order=tuple(order),
                    max_steps=max_steps, legacy_task_id=legacy.task_id, legacy_instruction=legacy.instruction, change=change)
