"""Static invariants of the 210-task registry and its tiers (no simulation, no rendering)."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import mujoco
import pytest

import afford_pro
from afford_pro.envs.base_franka_env import _load_model_with_portable_assets
from afford_pro.envs.parametric_env import ASSET_DIR
from afford_pro.task_definitions import ALL_210_TASKS, PRECISION_TIERS, load_all_augmented_prompts, precision_tier
from afford_pro.task_tiers import (AUDITED_MISMATCHES, TIERS, mismatch_reason, task_tier, tasks_in_tier,
                                   tier_counts_by_domain)


@lru_cache(maxsize=None)
def _model(scene: str) -> mujoco.MjModel:
    return _load_model_with_portable_assets(ASSET_DIR / scene)


def test_ids_and_numbers_are_unique_and_contiguous():
    ids = [t.task_id for t in ALL_210_TASKS]
    numbers = [t.task_number for t in ALL_210_TASKS]
    assert len(ids) == len(set(ids)) == 210
    assert sorted(numbers) == list(range(1, 211))
    assert afford_pro.list_tasks("legacy") == ids


def test_seven_domains_of_thirty():
    counts = Counter(t.domain for t in ALL_210_TASKS)
    assert len(counts) == 7 and set(counts.values()) == {30}


def test_frontier_tasks_are_not_core():
    assert not set(afford_pro.list_frontier_tasks()) & set(afford_pro.list_tasks("legacy"))


@pytest.mark.parametrize("task", ALL_210_TASKS, ids=lambda t: t.task_id)
def test_object_and_joint_exist_in_scene(task):
    model = _model(task.scene_xml)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task.object_name) >= 0, task.object_name
    if task.task_type != "press_button":
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, task.joint_name)
        assert joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE, task.joint_name
    for joint_name in (task.scene_bodies or {}):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0, joint_name


@pytest.mark.parametrize("task", ALL_210_TASKS, ids=lambda t: t.task_id)
def test_tolerances_follow_the_precision_tier(task):
    tol, tilt = precision_tier(task)
    assert (task.tolerance_xy, task.tolerance_z, task.tilt_tol_deg) == (tol, tol, tilt)
    assert task.domain in PRECISION_TIERS or task.domain.startswith("Long-Horizon")


@pytest.mark.parametrize("task", ALL_210_TASKS, ids=lambda t: t.task_id)
def test_instruction_text_is_clean(task):
    assert task.instruction.strip() == task.instruction and task.instruction
    assert not any(ord(c) < 32 for c in task.instruction), repr(task.instruction)


def test_every_task_has_grounding_flags():
    prompts = load_all_augmented_prompts()
    for task in ALL_210_TASKS:
        grounding = prompts[task.task_id]["sim_grounding"]
        assert isinstance(grounding["referent_in_scene"], bool)
        assert isinstance(grounding["instruction_covered_by_success"], bool)


def test_tiers_partition_the_registry():
    members = {tier: set(tasks_in_tier(tier)) for tier in TIERS}
    assert sum(len(m) for m in members.values()) == 210
    assert set().union(*members.values()) == set(afford_pro.list_tasks("legacy"))
    assert members["reach"] == {t.task_id for t in ALL_210_TASKS if t.task_type == "press_button"}


def test_chained_domains_have_no_core_tasks():
    for domain, counts in tier_counts_by_domain().items():
        if domain.startswith(("Long-Horizon", "Multi-Stage", "Sequential")):
            assert counts["core"] == 0, domain


def test_core_tasks_move_a_free_body():
    for task_id in tasks_in_tier("core"):
        assert task_tier(task_id) == "core"
        task = next(t for t in ALL_210_TASKS if t.task_id == task_id)
        assert task.task_type in ("pick_place", "insert") and task.min_lift > 0.0


def test_audited_mismatches_are_registered_and_explained():
    ids = {t.task_id for t in ALL_210_TASKS}
    assert set(AUDITED_MISMATCHES) <= ids
    for task_id in AUDITED_MISMATCHES:
        assert task_tier(task_id) == "instruction_mismatch"
    for task_id in tasks_in_tier("instruction_mismatch"):
        assert mismatch_reason(task_id)
    assert all(mismatch_reason(t) is None for t in tasks_in_tier("core") + tasks_in_tier("reach"))


def test_reported_core_subsets_are_core_tasks():
    from afford_pro.task_tiers import CORE_SUBSETS

    core = set(tasks_in_tier("core"))
    for name, members in CORE_SUBSETS.items():
        assert set(members) <= core, name
        assert len(set(members)) == len(members), name
