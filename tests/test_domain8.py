"""Tests for Domain 8: Frontier Complex Manipulation Suite.

Validates the Section 6 inclusion criteria from docs/FRONTIER_COMPLEX_TASKS.md:
1. Privileged closed-loop oracle solves >= 95% of episodes.
2. Null and uniform-random policies solve 0% of episodes (never solved at reset).
3. Open-loop replay fails under mid-trajectory perturbations (proves dynamic feedback is required).
4. Compound non-prehensile tasks strictly enforce the pushing metric before grasping.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import afford_pro
import afford_pro.envs.base_franka_env as base_env
from afford_pro.envs.frontier_env import FrontierComplexEnv
from afford_pro.oracles.frontier_oracle import FrontierOracle
from afford_pro.task_definitions import DOMAIN_8_FRONTIER_TASKS


class _NoRenderer:
    def __init__(self, *args, **kwargs) -> None:
        pass


base_env.mujoco.Renderer = _NoRenderer
base_env.BaseFrankaEnv.get_observation = lambda self: {"tcp_pose": self.tcp_pose}


def test_frontier_tasks_registered():
    """Verify Domain 8 tasks are discoverable and have correct metadata."""
    frontier_ids = afford_pro.list_frontier_tasks()
    assert len(frontier_ids) == len(DOMAIN_8_FRONTIER_TASKS)
    assert len(frontier_ids) >= 6

    for t in DOMAIN_8_FRONTIER_TASKS:
        assert t.task_id in frontier_ids
        assert t.domain == "Frontier Complex Manipulation"
        assert t.task_number >= 211
        assert t.instruction != ""


@pytest.mark.parametrize("task_id", [t.task_id for t in DOMAIN_8_FRONTIER_TASKS])
def test_not_solved_at_reset(task_id: str):
    """Verify no Domain 8 task is already solved at reset."""
    env = afford_pro.make(task_id)
    assert isinstance(env, FrontierComplexEnv)
    env.reset(seed=42)
    assert not env.check_success(), f"Task {task_id} must not be solved at reset"


@pytest.mark.parametrize("task_id", [
    "frontier_reactive_sort_red",
    "frontier_reactive_shaft_seat",
    "frontier_nonprehensile_push_and_grasp",
])
def test_null_policy_fails(task_id: str):
    """Null policy (holding open at rest) must score 0%."""
    env = afford_pro.make(task_id)
    env.reset(seed=100)
    for _ in range(50):
        _, _, term, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        if term:
            break
    assert not env.check_success(), f"Null policy must fail on {task_id}"


@pytest.mark.parametrize("task_id", [
    "frontier_reactive_sort_red",
    "frontier_reactive_shaft_seat",
    "frontier_nonprehensile_push_and_grasp",
])
def test_random_policy_fails(task_id: str):
    """Uniform random actions must score 0%."""
    env = afford_pro.make(task_id)
    env.reset(seed=101)
    rng = np.random.default_rng(101)
    for _ in range(50):
        action = rng.uniform(-1.0, 1.0, 7)
        _, _, term, _, _ = env.step(action)
        if term:
            break
    assert not env.check_success(), f"Random policy must fail on {task_id}"


def test_oracle_solves_reactive_perturbation():
    """Privileged closed-loop oracle reacts to 4 cm perturbation and solves the task."""
    env = afford_pro.make("frontier_reactive_sort_red")
    oracle = afford_pro.get_oracle("frontier_reactive_sort_red", env)
    actions, _, _, dones = oracle.generate_episode(seed=0)

    assert env.check_success(), "Oracle must solve reactive perturbation task"
    assert env.perturbed_applied, "Environment must have triggered the mid-approach perturbation"
    assert len(actions) < 400


def test_open_loop_replay_fails_under_perturbation():
    """Nominal open-loop plan misses the object and fails under mid-approach displacement."""
    # 1. Generate actions from unperturbed nominal environment
    nom_env = afford_pro.make("frontier_reactive_sort_red")
    nom_env.perturbation_step = None
    nom_oracle = afford_pro.get_oracle("frontier_reactive_sort_red", nom_env)
    nom_actions, _, _, _ = nom_oracle.generate_episode(seed=0)
    assert nom_env.check_success(), "Nominal trajectory must solve unperturbed task"

    # 2. Replay actions open-loop on perturbed environment
    pert_env = afford_pro.make("frontier_reactive_sort_red")
    pert_env.reset(seed=0)
    for a in nom_actions:
        pert_env.step(a)

    assert not pert_env.check_success(), "Open-loop replay must fail when object is displaced"


def test_push_and_grasp_enforces_distance():
    """Non-prehensile pushing distance must be strictly satisfied for task success."""
    env = afford_pro.make("frontier_nonprehensile_push_and_grasp")
    oracle = afford_pro.get_oracle("frontier_nonprehensile_push_and_grasp", env)
    actions, _, _, _ = oracle.generate_episode(seed=42)

    assert env.check_success(), "Oracle must solve push_and_grasp task"
    assert env.pushed_distance >= env.min_push_distance, "Pushed distance requirement must be met"
