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


def test_family_structure_matches_specification():
    """Verify Domain 8 has exactly 30 tasks partitioned into the 4 charter families:
    Family A: Tasks 211-218 (8 tasks, tool-mediated)
    Family B: Tasks 219-226 (8 tasks, non-prehensile pushing/pivoting)
    Family C: Tasks 227-233 (7 tasks, twist-lock & detents)
    Family D: Tasks 234-240 (7 tasks, reactive perturbations)
    """
    assert len(DOMAIN_8_FRONTIER_TASKS) == 30
    task_numbers = [t.task_number for t in DOMAIN_8_FRONTIER_TASKS]
    assert task_numbers == list(range(211, 241))

    fam_a = [t for t in DOMAIN_8_FRONTIER_TASKS if 211 <= t.task_number <= 218]
    fam_b = [t for t in DOMAIN_8_FRONTIER_TASKS if 219 <= t.task_number <= 226]
    fam_c = [t for t in DOMAIN_8_FRONTIER_TASKS if 227 <= t.task_number <= 233]
    fam_d = [t for t in DOMAIN_8_FRONTIER_TASKS if 234 <= t.task_number <= 240]

    assert len(fam_a) == 8
    assert len(fam_b) == 8
    assert len(fam_c) == 7
    assert len(fam_d) == 7
    assert all(t.task_type == "tool_use" for t in fam_a)
    assert all(t.task_type in ("push_overhang", "push_and_grasp") for t in fam_b)
    assert all(t.task_type == "twist_lock" for t in fam_c)
    assert all(t.task_type == "reactive_perturbation" for t in fam_d)


def test_push_and_grasp_enforces_distance():
    """Non-prehensile pushing distance must be strictly satisfied for task success."""
    env = afford_pro.make("frontier_nonprehensile_push_and_grasp")
    oracle = afford_pro.get_oracle("frontier_nonprehensile_push_and_grasp", env)
    actions, _, _, _ = oracle.generate_episode(seed=42)

    assert env.check_success(), "Oracle must solve push_and_grasp task"
    assert env.pushed_distance >= env.min_push_distance, "Pushed distance requirement must be met"


def test_push_overhang_success():
    """Verify non-prehensile overhang push (Task 219) moves plate to target overhang without dropping it."""
    env = afford_pro.make("frontier_push_thin_plate_overhang")
    oracle = afford_pro.get_oracle("frontier_push_thin_plate_overhang", env)
    actions, _, _, _ = oracle.generate_episode(seed=42)

    assert env.check_success(), "Oracle must solve push_thin_plate_overhang task"
    assert env.pushed_distance >= env.min_push_distance, "Overhang pushing distance must exceed threshold"


def test_double_displacement_recovery():
    """Verify closed-loop oracle recovers from two sequential mid-trajectory perturbations (Task 240)."""
    env = afford_pro.make("frontier_reactive_double_displacement")
    oracle = afford_pro.get_oracle("frontier_reactive_double_displacement", env)
    actions, _, _, _ = oracle.generate_episode(seed=0)

    assert env.check_success(), "Closed-loop oracle must solve double displacement task"
    assert env.perturbed_applied, "First perturbation must have fired"
    assert env.second_perturbed_applied, "Second perturbation must have fired"


def test_goal_fixture_shift_recovery():
    """Verify closed-loop oracle adapts when the goal fixture shifts mid-transport (Task 238)."""
    env = afford_pro.make("frontier_reactive_fixture_shift")
    oracle = afford_pro.get_oracle("frontier_reactive_fixture_shift", env)
    actions, _, _, _ = oracle.generate_episode(seed=0)

    assert env.check_success(), "Closed-loop oracle must adapt to dynamic goal shift"
    assert env.goal_shift_applied, "Goal fixture shift must have been triggered"


# ----------------------------------------------------------------- Family C: twist-lock and detent mechanisms
from afford_pro.envs.frontier_env import TWIST_LOCK_MECHANISMS, TwistLockEnv  # noqa: E402
from afford_pro.oracles.twist_lock_oracle import TwistLockOracle  # noqa: E402

TWIST_TASKS = [t.task_id for t in DOMAIN_8_FRONTIER_TASKS if t.task_type == "twist_lock"]
TWIST_SEEDS = 4  # per task; scripts/twist_lock_probe.py runs the full 21-seed measurement


def test_family_c_registered():
    """Tasks 227-233 are registered with the twist-lock env and oracle."""
    numbers = sorted(t.task_number for t in DOMAIN_8_FRONTIER_TASKS if t.task_type == "twist_lock")
    assert numbers == list(range(227, 234))
    for task_id in TWIST_TASKS:
        env = afford_pro.make(task_id)
        assert isinstance(env, TwistLockEnv)
        assert isinstance(afford_pro.get_oracle(task_id, env), TwistLockOracle)
        assert env.mechanism in TWIST_LOCK_MECHANISMS


def test_twist_lock_scene_is_contact_only():
    """No welds, mocap bodies, or snapping: the only equality constraint is the Panda finger coupling, and every
    mechanism degree of freedom is a native slide or hinge joint (springs as joint stiffness)."""
    import mujoco

    env = afford_pro.make("frontier_cap_push_turn_open")
    m = env.model
    assert m.neq == 1 and m.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert m.nmocap == 0
    springs = {"cap_press": 2000.0, "bayonet_press": 1500.0, "bayonet2_press": 1500.0}
    for name, k in springs.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
        assert m.jnt_stiffness[j] == pytest.approx(k)
    for name in ("cap_ring_hinge", "cap_spin", "bayonet_twist", "lock_plug_hinge", "lock_sleeve_hinge"):
        assert m.jnt_type[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)] == mujoco.mjtJoint.mjJNT_HINGE


def test_axial_force_is_spring_force():
    """F_z is read from the slide-joint spring force, k_c * dz."""
    env = afford_pro.make("frontier_cap_push_turn_open")
    env.reset(seed=0)
    env.data.qpos[env.press_qadr] = -0.011
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    assert env.axial_force == pytest.approx(2000.0 * 0.011)


@pytest.mark.parametrize("task_id", TWIST_TASKS)
def test_twist_lock_null_and_random_fail(task_id: str):
    """Null and uniform-random policies never satisfy Phi_C."""
    env = afford_pro.make(task_id)
    for policy in ("null", "random"):
        env.reset(seed=7)
        rng = np.random.default_rng(7)
        for _ in range(150):
            action = rng.uniform(-1.0, 1.0, 7) if policy == "random" else np.array([0, 0, 0, 0, 0, 0, 1.0])
            _, _, term, _, _ = env.step(action)
            assert not term, f"{policy} policy solved {task_id}"
        assert not env.check_success()


def test_twist_lock_oracle_success_rate():
    """The privileged oracle solves >= 95% of Family C episodes (TWIST_SEEDS reset seeds per task)."""
    outcomes = []
    for task_id in TWIST_TASKS:
        env = afford_pro.make(task_id)
        oracle = afford_pro.get_oracle(task_id, env)
        for seed in range(TWIST_SEEDS):
            oracle.generate_episode(seed=seed)
            outcomes.append(env.check_success())
            if env.mech.kind == "cap" and env.check_success():
                assert env.low_force_travel <= np.radians(5.0)
                assert env.forced_travel >= np.radians(170.0)  # the ring was turned under >= 20 N
    rate = float(np.mean(outcomes))
    assert rate >= 0.95, f"oracle success {rate:.1%} over {len(outcomes)} episodes"


@pytest.mark.parametrize("task_id", ["frontier_cap_push_turn_open", "frontier_bayonet_lock"])
def test_under_threshold_press_cannot_turn(task_id: str):
    """Turning with an axial press below the threshold (5 mm commanded: nominally 10 N on the cap, 7.5 N on the bayonet) leaves
    the mechanism locked: the cap freewheels over the ratchet and the bayonet pin stops against the lock ceiling."""
    env = afford_pro.make(task_id)
    oracle = TwistLockOracle(env, press_depth=0.005)
    oracle.generate_episode(seed=0)
    assert not env.check_success()
    start = env.mech.start_angle
    assert abs(env.mechanism_angle - start) < np.radians(10.0), "mechanism turned without the required preload"
    if env.mech.kind == "cap":
        assert env.peak_axial_force < 20.0
        assert env.forced_travel == 0.0
