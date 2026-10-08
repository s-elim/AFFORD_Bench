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
# Largest initial (x, y) TCP displacement from L1 jitter: +-1.5 cm per axis (PerturbationConfig).
MAX_JITTER_XY = 0.015 * math.sqrt(2.0)
# Largest initial (x, y) body displacement from L1 scatter: +-7 cm per axis (PerturbationConfig).
MAX_SCATTER_XY = 0.07 * math.sqrt(2.0)
MARGIN = 0.005
# Goal tolerance tiers (xy and z in m, tilt in deg) by domain, as specified for the hardened benchmark.
TIERS = {
    "Industrial & Manufacturing": (0.002, 5.0),           # D1
    "Culinary & Kitchen Operations": (0.008, 15.0),       # D2
    "Tabletop & Domestic": (0.008, 15.0),                 # D3
    "Laboratory & Life Sciences": (0.002, 5.0),           # D4
    "Multi-Stage Precision Chaining": (0.0025, 5.0),      # D6
    "Sequential Service & Culinary Assembly": (0.008, 15.0),  # D7
}
DISTRACTOR_DOMAINS = {"Industrial & Manufacturing", "Tabletop & Domestic", "Sequential Service & Culinary Assembly"}


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
    assert d_xy >= task.tolerance_xy + MAX_SCATTER_XY + MARGIN or d_z >= task.tolerance_z + MARGIN, (
        f"start-goal separation xy={d_xy:.3f} m, z={d_z:.3f} m is inside the success region after jitter"
    )


@pytest.mark.parametrize("task", [t for t in TASKS if t.task_type == "press_button"], ids=lambda t: t.task_id)
def test_press_goal_away_from_start_tcp(task) -> None:
    start = task.init_tcp_pos if task.init_tcp_pos is not None else READY_TCP
    assert math.dist(start, task.target_pos) >= task.tolerance_xy + MAX_JITTER_XY + MARGIN


def test_lift_tasks_have_positive_lift() -> None:
    for t in TASKS:
        assert t.min_lift >= 0.0
        if t.nominal_pos == t.target_pos:
            assert t.min_lift > 0.0, t.task_id


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_precision_tier(task) -> None:
    if task.domain in TIERS:
        tol, tilt = TIERS[task.domain]
    else:  # D5: tier of the scene (precise industrial and lab scenes, coarse kitchen and tabletop scenes)
        precise = task.scene_xml in ("industrial_assembly.xml", "lab_science.xml")
        tol, tilt = (0.002, 5.0) if precise else (0.008, 15.0)
    assert (task.tolerance_xy, task.tolerance_z, task.tilt_tol_deg) == (tol, tol, tilt)


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_distractors_by_domain(task) -> None:
    assert task.distractors == (task.domain in DISTRACTOR_DOMAINS)


@pytest.mark.parametrize("task", [t for t in TASKS if t.task_type != "press_button"], ids=lambda t: t.task_id)
def test_manipulation_requires_lift(task) -> None:
    """No push-only solutions: every manipulated body must be lifted at least 2 cm before it can succeed."""
    assert task.min_lift >= 0.02


def test_perturbation_ranges() -> None:
    pytest.importorskip("mujoco")
    sys.path.insert(0, str(ROOT))
    from afford_pro.perturbations.engine import PerturbationConfig

    c = PerturbationConfig()
    assert (c.nominal_xy_m, c.nominal_yaw_deg, c.nominal_tilt_deg) == (0.07, 45.0, 10.0)
    assert c.friction_range == (0.12, 0.25)
    assert c.mass_scale_range == (1.8, 2.5)
