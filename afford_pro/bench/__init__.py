"""AFFORD-Bench tasks: the 210 tasks, built so that every instruction names simulated objects and every success check
covers the whole instruction (docs/roadmap/T9_TASK_REDESIGN.md).

    import afford_pro
    env = afford_pro.make("kitchen_cheese_on_patty")      # or afford_pro.bench.make_env(32)
    oracle = afford_pro.get_oracle("kitchen_cheese_on_patty", env)
"""
from __future__ import annotations

from typing import Any


def get_spec(task: str | int):
    """The TaskSpec of a task, by id or number."""
    from afford_pro.bench.tasks import TASK_BY_NUMBER, TASK_MAP

    if isinstance(task, int) or (isinstance(task, str) and task.isdigit()):
        number = int(task)
        if number not in TASK_BY_NUMBER:
            raise KeyError(f"no task number {number}")
        return TASK_BY_NUMBER[number]
    if task not in TASK_MAP:
        raise KeyError(f"Task '{task}' not found among the AFFORD-Bench tasks ({len(TASK_MAP)} tasks).")
    return TASK_MAP[task]


def make_env(task: str | int, **kwargs: Any):
    from afford_pro.bench.env import GoalFrankaEnv

    return GoalFrankaEnv(get_spec(task), **kwargs)


def make_oracle(env: Any):
    from afford_pro.bench.oracle import GoalOracle

    return GoalOracle(env)


def list_tasks() -> list[str]:
    from afford_pro.bench.tasks import TASKS

    return [t.task_id for t in TASKS]
