"""The 210 AFFORD-Bench tasks and their static consistency checks.

Each task replaces the legacy task with the same number. `static_problems()` lists every way a specification can be
inconsistent with its scene without simulating it; the runtime checks (attainable goals, valid starts, oracle success)
are in scripts/validate_bench_tasks.py.
"""
from __future__ import annotations

from afford_pro.bench.scenes import SCENES
from afford_pro.bench.spec import (TERMINAL_GOALS, Closed, GripperAbove, HeldOver, On, Opened, Placed, Pressed, Slid,
                                Slots, TaskSpec, Touching)
from afford_pro.bench.tasks import (d1_industrial, d2_kitchen, d3_tabletop, d4_lab, d5_long_horizon,
                                 d6_precision_chaining, d7_service)

TASKS: list[TaskSpec] = sorted(
    d1_industrial.TASKS + d2_kitchen.TASKS + d3_tabletop.TASKS + d4_lab.TASKS + d5_long_horizon.TASKS
    + d6_precision_chaining.TASKS + d7_service.TASKS, key=lambda t: t.number)
TASK_MAP: dict[str, TaskSpec] = {t.task_id: t for t in TASKS}
TASK_BY_NUMBER: dict[int, TaskSpec] = {t.number: t for t in TASKS}


def _goal_names(spec: TaskSpec, goal) -> list[tuple[str, str]]:
    """(kind, name) of every scene entity a goal refers to."""
    if isinstance(goal, (On, Slid, HeldOver)):
        return [("body", goal.body), ("place", goal.target)]
    if isinstance(goal, GripperAbove):
        return [("place", goal.target)]
    if isinstance(goal, Pressed):
        return [("button", goal.button)]
    if isinstance(goal, (Closed, Opened)):
        return [("slider", goal.slider)]
    return [("body", goal.body)]


def _fixed_slot_starts(spec: TaskSpec, scene) -> list[str]:
    """Slot starts that collapse to one slot: a start in a goal that no other goal precedes is rejected at reset
    (a goal that follows another, as in "... then put it back", may hold at the start)."""
    out = []
    first = [g for k, g in enumerate(spec.goals) if not any(j == k for _, j in spec.order)]
    for body in spec.bodies_moved():
        start = spec.starts.get(body, scene.bodies[body].home if body in scene.bodies else None)
        if not isinstance(start, Slots):
            continue
        goal_targets = {g.target for g in first if isinstance(g, (On, Slid)) and g.body == body}
        usable = [t for t in start.targets if t not in goal_targets]
        if len(usable) < 2:
            out.append(f"{body} has one usable slot ({', '.join(usable) or 'none'}): its other slots are its goal, "
                       f"so its start would be fixed")
    return out


def _spec_problems(spec: TaskSpec) -> list[str]:
    scene = SCENES.get(spec.scene)
    if scene is None:
        return [f"unknown scene {spec.scene}"]
    known = {"body": set(scene.bodies), "place": set(scene.targets) | set(scene.bodies),
             "button": set(scene.buttons), "slider": set(scene.sliders)}
    out = [f"{kind} {name!r} not in {spec.scene}" for g in spec.goals for kind, name in _goal_names(spec, g)
           if name not in known[kind]]
    for body, start in spec.starts.items():
        if body not in scene.bodies:
            out.append(f"start for unknown body {body!r}")
        targets = start.targets if isinstance(start, Slots) else (start.target,) if isinstance(start, Placed) else ()
        out += [f"start target {t!r} not in {spec.scene}" for t in targets if t not in known["place"]]
    out += [f"slider {s!r} not in {spec.scene}" for s in spec.sliders if s not in scene.sliders]
    for body, start in spec.starts.items():
        target = getattr(start, "target", None)
        if target in scene.attached and spec.sliders:
            out.append(f"{body} starts in {target}, which moves with a mechanism: keep the scene's slider positions")
    out += _fixed_slot_starts(spec, scene)
    for k, g in enumerate(spec.goals):
        if isinstance(g, TERMINAL_GOALS) and not getattr(g, "once", False) and k != len(spec.goals) - 1:
            out.append(f"goal {k} ({type(g).__name__}) must be held to the end, so it must be the last goal")
    for k, g in enumerate(spec.goals[:-1]):
        later = [h for h in spec.goals[k + 1:] if getattr(h, "body", None) == getattr(g, "body", object())]
        if isinstance(g, (On, HeldOver)) and later and not g.once and any(isinstance(h, (On, HeldOver)) for h in later):
            out.append(f"goal {k} places {g.body} but a later goal moves it again: mark it once=True")
    n = len(spec.goals)
    out += [f"order {pair} refers to a missing goal" for pair in spec.order
            if not (0 <= pair[0] < n and 0 <= pair[1] < n and pair[0] != pair[1])]
    if not spec.instruction.endswith("."):
        out.append("instruction must be a sentence ending with a period")
    return out


def static_problems() -> dict[str, list[str]]:
    """Problems per task id (empty dict when every specification is consistent)."""
    problems: dict[str, list[str]] = {}
    numbers = [t.number for t in TASKS]
    if sorted(numbers) != list(range(1, 211)):
        problems["<registry>"] = ["task numbers are not exactly 1..210"]
    if len(TASK_MAP) != len(TASKS):
        problems.setdefault("<registry>", []).append("duplicate task ids")
    seen: dict[tuple, str] = {}
    for spec in TASKS:
        found = _spec_problems(spec)
        key = (spec.scene, spec.goals, tuple(sorted((b, repr(s)) for b, s in spec.starts.items())))
        if key in seen:
            found.append(f"same scene, start and goals as {seen[key]}")
        seen.setdefault(key, spec.task_id)
        if found:
            problems[spec.task_id] = found
    return problems
