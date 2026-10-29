"""Unit tests for AFFORD-Bench Paired Base/Dynamic Evaluation Suite and Force Safety."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import afford_pro
from afford_pro.dynamic_suite import (
    DISPLACEMENT_DIRECTIONS,
    PANDA_TORQUE_LIMITS,
    MAX_SAFE_CONTACT_FORCE,
    build_paired_dynamic_suite,
    run_paired_case,
    summarize_paired_outcomes,
)
from afford_pro.envs.headless import headless


def test_suite_generation():
    """Verify deterministic paired case generation across representative tasks."""
    cases = build_paired_dynamic_suite(
        task_ids=["industrial_shaft_assembly", "kitchen_patty_on_bun"],
        seeds=[42, 101],
        directions=["+x", "-y"],
    )
    assert len(cases) == 2 * 2 * 2  # 2 tasks * 2 seeds * 2 directions = 8 cases
    assert cases[0].task_id == "industrial_shaft_assembly"
    assert cases[0].displacement_direction == "+x"
    assert cases[0].displacement_offset == [0.03, 0.0, 0.0]
    assert cases[0].seed == 42


def test_env_apply_object_displacement():
    """Verify apply_object_displacement moves the target object and zeroes velocity."""
    with headless():
        env = afford_pro.make("tabletop_sort_red", task_set="legacy")
        try:
            obs, _ = env.reset(seed=42)

            target_qposadr = env.target_qposadr
            assert target_qposadr is not None

            initial_pos = env.data.qpos[target_qposadr : target_qposadr + 3].copy()
            offset = np.array([0.03, -0.02, 0.0])

            applied = env.apply_object_displacement(offset)
            assert applied is True
            assert env.dynamic_perturbed_applied is True

            new_pos = env.data.qpos[target_qposadr : target_qposadr + 3]
            np.testing.assert_allclose(new_pos, initial_pos + offset, atol=1e-5)

            dof = env.free_qpos_to_dof[target_qposadr]
            np.testing.assert_allclose(env.data.qvel[dof : dof + 6], 0.0, atol=1e-5)
        finally:
            env.close()


def test_force_and_torque_safety_tracking():
    """Verify peak force tracking and torque saturation bounds in BaseFrankaEnv."""
    with headless():
        env = afford_pro.make("industrial_shaft_assembly", task_set="legacy")
        try:
            obs, _ = env.reset(seed=42)

            assert hasattr(env, "peak_contact_force")
            assert hasattr(env, "force_limit_exceeded")
            assert hasattr(env, "torque_limit_exceeded")
            assert hasattr(env, "safety_violation")
            assert env.peak_contact_force == 0.0
            assert not env.safety_violation

            # Run a step
            obs, rew, term, trunc, info = env.step(np.zeros(7))
            assert "peak_contact_force" in info
            assert "force_limit_exceeded" in info
            assert "torque_limit_exceeded" in info
            assert "safety_violation" in info
            assert "dynamic_perturbed" in info

            # Verify Panda torque limit array
            np.testing.assert_allclose(env.panda_torque_limits, PANDA_TORQUE_LIMITS)
            assert env.max_safe_contact_force == MAX_SAFE_CONTACT_FORCE
        finally:
            env.close()


def test_paired_run_execution():
    """Execute a single matched Base/Dynamic pair and verify prefix matching and outcome."""
    cases = build_paired_dynamic_suite(
        task_ids=["tabletop_sort_red"],
        seeds=[100],
        directions=["+x"],
        trigger_type="mid_approach_step",
    )
    assert len(cases) == 1
    case = cases[0]
    case.trigger_step = 10  # short trigger step for rapid test

    outcome = run_paired_case(case, policy_fn=None, use_headless=True)

    assert outcome.case_id == case.case_id
    assert outcome.trigger_step == 10
    assert outcome.event_triggered is True
    assert outcome.category in ("preserved_success", "event_regression", "recovery", "persistent_failure")
    assert isinstance(outcome.base_peak_force, float)
    assert isinstance(outcome.dynamic_peak_force, float)
    assert isinstance(outcome.compliant_dynamic_success, bool)


def test_open_loop_fails_under_displacement():
    """Verify that open-loop replay of nominal actions fails when the object is displaced."""
    with headless():
        env = afford_pro.make("tabletop_sort_red", task_set="legacy")
        try:
            obs, _ = env.reset(seed=101)

            # 1. Collect nominal actions from privileged oracle
            oracle = afford_pro.get_oracle("tabletop_sort_red", env)
            actions, _, _, _ = oracle.generate_episode(seed=101)
            assert env.check_success(), "Oracle must solve unperturbed task"

            # 2. Replay open-loop with mid-trajectory displacement at step 20
            env.reset(seed=101)
            for step_i, a in enumerate(actions):
                if step_i == 20:
                    env.apply_object_displacement(np.array([0.04, 0.0, 0.0]))
                env.step(a)

            # Open-loop replay must fail because the arm reaches for where the object used to be
            assert not env.check_success(), "Open-loop replay must fail under mid-trajectory displacement"
        finally:
            env.close()
