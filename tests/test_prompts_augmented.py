"""Consistency checks between configs/prompts_augmented.yaml and afford_pro/task_definitions.py."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "configs" / "prompts_augmented.yaml"
TIERS = ("geometric_spatial", "affordance_physical", "colloquial_terse", "atomic_subtasks")
PREDICATE = re.compile(r"^[a-z_0-9]+(\([a-z_0-9]+(, [a-z_0-9.]+)*\))?$")


def _load_task_defs():
    # Load the module by path so the test does not import MuJoCo through afford_pro/__init__.py.
    spec = importlib.util.spec_from_file_location("afford_task_definitions", ROOT / "afford_pro" / "task_definitions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ALL_210_TASKS


def _fmt(p: list[float]) -> str:
    return f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m"


TASK_DEFS = _load_task_defs()
DATA = yaml.safe_load(PROMPTS.read_text())
ENTRIES = DATA["tasks"]
PAIRS = list(zip(TASK_DEFS, ENTRIES))


def test_covers_all_tasks_in_order() -> None:
    assert DATA["num_tasks"] == len(ENTRIES) == len(TASK_DEFS) == 210
    assert [e["task_number"] for e in ENTRIES] == list(range(1, 211))
    assert [e["task_id"] for e in ENTRIES] == [t.task_id for t in TASK_DEFS]


@pytest.mark.parametrize("task_def, entry", PAIRS, ids=[t.task_id for t in TASK_DEFS])
def test_entry_matches_definition(task_def, entry) -> None:
    g = entry["sim_grounding"]
    assert entry["task_type"] == task_def.task_type
    assert entry["nominal_instruction"] == task_def.instruction
    assert g["object"] == task_def.object_name
    assert g["scene"] == task_def.scene_xml
    assert g["start_xyz_m"] == pytest.approx(task_def.nominal_pos)
    assert g["goal_xyz_m"] == pytest.approx(task_def.target_pos)
    assert g["tolerance_xy_m"] == pytest.approx(task_def.tolerance_xy)
    assert g["tolerance_z_m"] == pytest.approx(task_def.tolerance_z)
    assert g["tilt_tolerance_deg"] == pytest.approx(task_def.tilt_tol_deg)
    assert g["min_lift_m"] == pytest.approx(task_def.min_lift)
    assert g["distractors"] == task_def.distractors

    geo = entry["prompts"]["geometric_spatial"]
    assert _fmt(task_def.target_pos) in geo
    if task_def.task_type != "press_button":
        assert _fmt(task_def.nominal_pos) in geo


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["task_id"] for e in ENTRIES])
def test_prompt_tiers_well_formed(entry) -> None:
    prompts = entry["prompts"]
    assert set(prompts) == set(TIERS)
    for tier in TIERS[:3]:
        assert isinstance(prompts[tier], str) and prompts[tier].strip()
    assert len(prompts["colloquial_terse"].split()) <= 8

    steps = prompts["atomic_subtasks"]
    assert 2 <= len(steps) <= 3
    assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
    for s in steps:
        assert s["instruction"].strip()
        assert s["pre"] and s["post"]
        for p in s["pre"] + s["post"]:
            assert PREDICATE.match(p), p


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["task_id"] for e in ENTRIES])
def test_terminal_step_is_first_to_satisfy_success(entry) -> None:
    g = entry["sim_grounding"]
    steps = entry["prompts"]["atomic_subtasks"]
    hits = [s["step"] for s in steps if all(p in s["post"] for p in g["terminal_predicates"])]
    assert hits, "no subtask reaches the simulator's success predicate"
    assert g["episode_terminates_after_subtask"] == hits[0]


def test_flags_are_consistent() -> None:
    for e in ENTRIES:
        g = e["sim_grounding"]
        if e["task_number"] > 120 or not g["referent_in_scene"]:
            assert g["instruction_covered_by_success"] is False


def test_no_dashes_in_text() -> None:
    text = PROMPTS.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text
