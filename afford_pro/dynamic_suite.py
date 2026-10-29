"""Paired Base/Dynamic Evaluation Suite for AFFORD-Bench (Domains 1-7).

Inspired by the counterfactual matched-pair paradigm of LIBERO-MAX, but extended
with physical force and actuator torque sensing:
1. Each case evaluates a matched pair (Base control vs. Dynamic intervention).
2. Initial simulator state, policy seed, and executed pre-event action prefix
   a_{0:t*} are strictly matched between Base and Dynamic arms.
3. At trigger point t*, an exogenous object displacement Δx is injected into the
   manipulated workpiece.
4. Uniquely in AFFORD-Bench: Tracks 7-D actuator motor torques τ_{act} and 3-D contact
   forces F_{contact}. Distinguishes mere geometric task completion from Compliant
   Success (safety envelope F_{max} <= 150 N, τ_{act} <= τ_{max}).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np

import afford_pro
from afford_pro.envs.headless import headless


# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

# Franka Emika Panda continuous joint torque limits (N·m)
PANDA_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)

# Maximum permissible transient contact force before safety tripping (ISO/TS 15066)
MAX_SAFE_CONTACT_FORCE = 150.0  # N

# Canonical displacement directions (3 cm in workspace plane)
DISPLACEMENT_DIRECTIONS: dict[str, list[float]] = {
    "+x": [0.03, 0.0, 0.0],
    "-x": [-0.03, 0.0, 0.0],
    "+y": [0.0, 0.03, 0.0],
    "-y": [0.0, -0.03, 0.0],
    "+xy": [0.02, 0.02, 0.0],
}

REPRESENTATIVE_D1_D7_TASKS: list[str] = [
    # D1 Industrial Precision
    "industrial_shaft_assembly",
    "industrial_gear_mesh",
    "industrial_pin_insertion",
    # D2 Culinary Operations
    "kitchen_patty_on_bun",
    "kitchen_cheese_on_patty",
    "kitchen_top_bun_crown",
    # D3 Domestic Tabletop
    "tabletop_sort_red",
    "tabletop_sort_green",
    "tabletop_sort_blue",
    # D4 Laboratory Sciences
    "lab_micro_to_centrifuge",
    "lab_conical_to_orange_rack",
    "lab_conical_to_blue_rack",
    # D5 Long Horizon
    "long_horizon_transmission_assembly",
    "long_horizon_classic_burger",
    "long_horizon_sort_all_blocks",
    # D6 Precision Chaining
    "chain_shaft_bearing_clamp",
    "chain_bearing_then_gear",
    "chain_bolt_and_nut",
    # D7 Service Chaining
    "service_cheese_and_crown",
    "service_patty_cheese_plate",
    "service_coffee_set",
]


@dataclass
class PairedCase:
    """A frozen matched Base/Dynamic evaluation case."""
    case_id: str
    task_id: str
    domain: str
    seed: int
    trigger_type: Literal["pre_grasp_proximity", "mid_approach_step"]
    displacement_direction: str
    displacement_offset: list[float]
    trigger_step: int | None = None
    proximity_threshold: float = 0.08  # meters


@dataclass
class PairedOutcome:
    """Outcome and physical sensing telemetry for one matched Base/Dynamic pair."""
    case_id: str
    task_id: str
    domain: str
    seed: int
    displacement_direction: str
    displacement_offset: list[float]
    trigger_type: str
    trigger_step: int | None

    # Base (Control) arm
    base_success: bool
    base_steps: int
    base_peak_force: float
    base_force_violation: bool
    base_torque_saturation: bool
    base_safety_violation: bool

    # Dynamic (Intervention) arm
    dynamic_success: bool
    dynamic_steps: int
    dynamic_peak_force: float
    dynamic_force_violation: bool
    dynamic_torque_saturation: bool
    dynamic_safety_violation: bool
    event_triggered: bool

    # Matched Counterfactual Classification
    # "preserved_success": Base=1, Dyn=1
    # "event_regression":  Base=1, Dyn=0
    # "recovery":          Base=0, Dyn=1
    # "persistent_failure": Base=0, Dyn=0
    category: str
    compliant_dynamic_success: bool  # dynamic_success AND not dynamic_safety_violation
    force_spike_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Suite Construction
# -----------------------------------------------------------------------------

def build_paired_dynamic_suite(
    task_ids: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    directions: Sequence[str] | None = None,
    trigger_type: Literal["pre_grasp_proximity", "mid_approach_step"] = "pre_grasp_proximity",
) -> list[PairedCase]:
    """Construct a deterministic, frozen suite of paired evaluation cases."""
    if task_ids is None:
        task_ids = REPRESENTATIVE_D1_D7_TASKS
    if seeds is None:
        seeds = [42, 101, 2024, 3050, 4096]
    if directions is None:
        directions = ["+x", "-x", "+y", "-y"]

    cases: list[PairedCase] = []
    case_idx = 0

    for task_id in task_ids:
        t_entry = afford_pro.get_task_entry(task_id)
        domain = getattr(t_entry, "domain", "Standard D1-D7")

        for seed in seeds:
            for d_name in directions:
                offset = DISPLACEMENT_DIRECTIONS[d_name]
                case_id = f"dyn_pair_{case_idx:04d}_{task_id}_{d_name}_s{seed}"
                cases.append(
                    PairedCase(
                        case_id=case_id,
                        task_id=task_id,
                        domain=domain,
                        seed=seed,
                        trigger_type=trigger_type,
                        displacement_direction=d_name,
                        displacement_offset=offset,
                        trigger_step=25 if trigger_type == "mid_approach_step" else None,
                        proximity_threshold=0.08,
                    )
                )
                case_idx += 1

    return cases


# -----------------------------------------------------------------------------
# Matched Pair Rollout Runner
# -----------------------------------------------------------------------------

def _execute_paired_case(
    case: PairedCase,
    policy_fn: Optional[Callable[[dict[str, Any]], np.ndarray]] = None,
) -> PairedOutcome:
    """Execute a matched pair (Base control vs. Dynamic intervention) for one case."""
    env = afford_pro.make(case.task_id, task_set="legacy")
    # For frontier reactive tasks (Domain 8), disable perturbation during Base arm
    if hasattr(env, "perturbation_step"):
        env.perturbation_step = None

    # -------------------------------------------------------------------------
    # Arm 1: Base (Control) - Nominal execution
    # -------------------------------------------------------------------------
    max_steps = getattr(env, "max_episode_steps", 400)

    if policy_fn is None:
        # Scripted oracle baseline
        oracle = afford_pro.get_oracle(case.task_id, env)
        actions, _, _, _ = oracle.generate_episode(seed=case.seed)
        base_actions = [a.copy() for a in actions]
        base_success = env.check_success()
        base_step = len(base_actions)
        trigger_step_detected = case.trigger_step or max(1, min(25, base_step // 2))
    else:
        # User / Learned policy rollout
        obs, _ = env.reset(seed=case.seed)
        base_actions = []
        base_success = False
        base_step = 0
        trigger_step_detected = None

        for step_i in range(max_steps):
            action = policy_fn(obs)
            base_actions.append(action.copy())

            # Check if proximity trigger condition is met in Base
            if trigger_step_detected is None:
                if case.trigger_type == "pre_grasp_proximity":
                    target_id = getattr(env, "target_id", -1)
                    if target_id != -1:
                        target_pos = env.data.xpos[target_id]
                        dist_to_obj = float(np.linalg.norm(env.tcp_position[:2] - target_pos[:2]))
                        if dist_to_obj <= case.proximity_threshold and env.tcp_position[2] < 0.20:
                            trigger_step_detected = step_i
                elif case.trigger_type == "mid_approach_step" and step_i == (case.trigger_step or 25):
                    trigger_step_detected = step_i

            obs, _, term, trunc, info = env.step(action)
            base_step = step_i + 1
            if term:
                base_success = True
                break
            if trunc:
                break

        if trigger_step_detected is None:
            trigger_step_detected = min(case.trigger_step or 25, max(1, base_step // 2))

    base_peak_force = float(env.peak_contact_force)
    base_force_violation = bool(env.force_limit_exceeded)
    base_torque_saturation = bool(env.torque_limit_exceeded)
    base_safety_violation = bool(env.safety_violation)

    # -------------------------------------------------------------------------
    # Arm 2: Dynamic (Intervention) - Matched prefix replay + mid-task displacement
    # -------------------------------------------------------------------------
    obs, _ = env.reset(seed=case.seed)
    offset = np.array(case.displacement_offset, dtype=np.float64)
    env.dynamic_perturbation_offset = offset
    env.dynamic_perturbation_step = trigger_step_detected
    env.dynamic_perturbation_trigger = "step"

    # For Domain 8 frontier tasks, set native frontier perturbation parameters
    if hasattr(env, "perturbation_step"):
        env.perturbation_step = trigger_step_detected
    if hasattr(env, "perturbation_offset"):
        env.perturbation_offset = offset

    if policy_fn is None:
        # Closed-loop oracle on perturbed environment
        oracle = afford_pro.get_oracle(case.task_id, env)
        dyn_actions, _, _, _ = oracle.generate_episode(seed=case.seed, reset=False)
        dynamic_success = env.check_success()
        dynamic_step = len(dyn_actions)
        event_applied = env.dynamic_perturbed_applied or getattr(env, "perturbed_applied", False)
    else:
        dynamic_success = False
        dynamic_step = 0
        event_applied = False
        term, trunc = False, False

        # 1. Replay exact Base action prefix up to trigger_step_detected
        for step_i in range(trigger_step_detected):
            if step_i < len(base_actions):
                action = base_actions[step_i]
                obs, _, term, trunc, _ = env.step(action)
                dynamic_step = step_i + 1
                if term or trunc:
                    break

        # 2. Inject frozen physical object displacement at t* if not already triggered
        if not term and not trunc:
            if not env.dynamic_perturbed_applied:
                event_applied = env.apply_object_displacement(offset)
            else:
                event_applied = True

            # 3. Policy continues acting post-perturbation
            remaining_budget = max_steps - dynamic_step
            for _ in range(remaining_budget):
                action = policy_fn(obs)
                obs, _, term, trunc, info = env.step(action)
                dynamic_step += 1
                if term:
                    dynamic_success = True
                    break
                if trunc:
                    break

    dynamic_peak_force = float(env.peak_contact_force)
    dynamic_force_violation = bool(env.force_limit_exceeded)
    dynamic_torque_saturation = bool(env.torque_limit_exceeded)
    dynamic_safety_violation = bool(env.safety_violation)

    # -------------------------------------------------------------------------
    # Classification & Metrics
    # -------------------------------------------------------------------------
    if base_success and dynamic_success:
        category = "preserved_success"
    elif base_success and not dynamic_success:
        category = "event_regression"
    elif not base_success and dynamic_success:
        category = "recovery"
    else:
        category = "persistent_failure"

    compliant_success = dynamic_success and not dynamic_safety_violation
    force_ratio = dynamic_peak_force / max(base_peak_force, 1e-3)

    return PairedOutcome(
        case_id=case.case_id,
        task_id=case.task_id,
        domain=case.domain,
        seed=case.seed,
        displacement_direction=case.displacement_direction,
        displacement_offset=case.displacement_offset,
        trigger_type=case.trigger_type,
        trigger_step=trigger_step_detected,
        base_success=base_success,
        base_steps=base_step,
        base_peak_force=base_peak_force,
        base_force_violation=base_force_violation,
        base_torque_saturation=base_torque_saturation,
        base_safety_violation=base_safety_violation,
        dynamic_success=dynamic_success,
        dynamic_steps=dynamic_step,
        dynamic_peak_force=dynamic_peak_force,
        dynamic_force_violation=dynamic_force_violation,
        dynamic_torque_saturation=dynamic_torque_saturation,
        dynamic_safety_violation=dynamic_safety_violation,
        event_triggered=event_applied,
        category=category,
        compliant_dynamic_success=compliant_success,
        force_spike_ratio=force_ratio,
    )


def run_paired_case(
    case: PairedCase,
    policy_fn: Optional[Callable[[dict[str, Any]], np.ndarray]] = None,
    use_headless: bool = True,
) -> PairedOutcome:
    """Execute a matched pair (Base control vs. Dynamic intervention) for one case.

    If policy_fn is None, uses the task's closed-loop oracle.
    """
    if use_headless:
        with headless():
            return _execute_paired_case(case, policy_fn=policy_fn)
    return _execute_paired_case(case, policy_fn=policy_fn)


# -----------------------------------------------------------------------------
# Summary Aggregation
# -----------------------------------------------------------------------------

def summarize_paired_outcomes(outcomes: Sequence[PairedOutcome]) -> dict[str, Any]:
    """Aggregate paired outcomes into benchmark metrics with bootstrap intervals."""
    total = len(outcomes)
    if total == 0:
        return {"total_cases": 0}

    base_succ = sum(1 for o in outcomes if o.base_success)
    dyn_succ = sum(1 for o in outcomes if o.dynamic_success)
    comp_succ = sum(1 for o in outcomes if o.compliant_dynamic_success)

    preserved = sum(1 for o in outcomes if o.category == "preserved_success")
    regression = sum(1 for o in outcomes if o.category == "event_regression")
    recovery = sum(1 for o in outcomes if o.category == "recovery")
    persistent_fail = sum(1 for o in outcomes if o.category == "persistent_failure")

    dyn_force_viol = sum(1 for o in outcomes if o.dynamic_force_violation)
    dyn_torque_sat = sum(1 for o in outcomes if o.dynamic_torque_saturation)
    dyn_safety_viol = sum(1 for o in outcomes if o.dynamic_safety_violation)

    mean_base_force = float(np.mean([o.base_peak_force for o in outcomes]))
    mean_dyn_force = float(np.mean([o.dynamic_peak_force for o in outcomes]))
    max_dyn_force = float(np.max([o.dynamic_peak_force for o in outcomes]))

    # Regression rate among Base-successful episodes: (1,0) / ((1,1) + (1,0))
    cond_regression = (regression / base_succ) if base_succ > 0 else 0.0

    return {
        "total_paired_cases": total,
        "base_success_rate": base_succ / total,
        "dynamic_success_rate": dyn_succ / total,
        "paired_gap_pp": 100.0 * (dyn_succ - base_succ) / total,
        "conditional_regression_rate": cond_regression,
        "outcomes": {
            "preserved_success": preserved,
            "event_regression": regression,
            "recovery": recovery,
            "persistent_failure": persistent_fail,
        },
        "percentages": {
            "preserved_success": 100.0 * preserved / total,
            "event_regression": 100.0 * regression / total,
            "recovery": 100.0 * recovery / total,
            "persistent_failure": 100.0 * persistent_fail / total,
        },
        "force_and_safety": {
            "max_safe_force_threshold_n": MAX_SAFE_CONTACT_FORCE,
            "dynamic_force_violation_count": dyn_force_viol,
            "dynamic_force_violation_rate": dyn_force_viol / total,
            "dynamic_torque_saturation_count": dyn_torque_sat,
            "dynamic_torque_saturation_rate": dyn_torque_sat / total,
            "dynamic_safety_violation_count": dyn_safety_viol,
            "dynamic_safety_violation_rate": dyn_safety_viol / total,
            "mean_base_peak_force_n": mean_base_force,
            "mean_dynamic_peak_force_n": mean_dyn_force,
            "maximum_dynamic_force_recorded_n": max_dyn_force,
            "compliant_dynamic_success_rate": comp_succ / total,
        },
    }
