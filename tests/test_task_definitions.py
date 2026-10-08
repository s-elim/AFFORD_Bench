"""Invariants on afford_pro/task_definitions.py that keep every task non-trivial at reset.

These checks need no MuJoCo; scripts/sanity_baselines.py measures the same properties in simulation.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# TCP position of the ready joint configuration (BaseFrankaEnv.ready_q), world frame (m).
READY_TCP = (0.364, 0.0, 0.505)
# Largest initial (x, y) displacement from L1 jitter: +-1.5 cm per axis (PerturbationConfig).
MAX_JITTER_XY = 0.015 * math.sqrt(2.0)
MARGIN = 0.005
PRESS_TOL = 0.025


def _load_task_defs():
    spec = importlib.util.spec_from_file_location("afford_task_definitions_inv", ROOT / "afford_pro" / "task_definitions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ALL_210_TASKS


TASKS = _load_task_defs()
IDS = [t.task_id for t in TASKS]


def test_registry_shape() -> None:
    assert len(TASKS) == 210
    assert len(set(IDS)) == 210
    assert [t.task_number for t in TASKS] == list(range(1, 211))


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_instruction_is_clean_text(task) -> None:
    assert "\t" not in task.instruction
    assert "\\" not in task.instruction and "$" not in task.instruction


@pytest.mark.parametrize("task", [t for t in TASKS if t.task_type != "press_button"], ids=lambda t: t.task_id)
def test_goal_not_satisfied_at_reset(task) -> None:
    """Either the goal lies outside the jittered start region, or success requires a lift."""
    if task.min_lift > 0.0:
        return
    d_xy = math.dist(task.nominal_pos[:2], task.target_pos[:2])
    d_z = abs(task.nominal_pos[2] - task.target_pos[2])
    assert d_xy >= task.tolerance_xy + MAX_JITTER_XY + MARGIN or d_z >= task.tolerance_z + MARGIN, (
        f"start-goal separation xy={d_xy:.3f} m, z={d_z:.3f} m is inside the success region after jitter"
    )


@pytest.mark.parametrize("task", [t for t in TASKS if t.task_type == "press_button"], ids=lambda t: t.task_id)
def test_press_goal_away_from_start_tcp(task) -> None:
    start = task.init_tcp_pos if task.init_tcp_pos is not None else READY_TCP
    assert math.dist(start, task.target_pos) >= PRESS_TOL + MAX_JITTER_XY + MARGIN


def test_lift_tasks_have_positive_lift() -> None:
    for t in TASKS:
        assert t.min_lift >= 0.0
        if t.nominal_pos == t.target_pos:
            assert t.min_lift > 0.0, t.task_id
