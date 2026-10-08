"""Frontier Complex Manipulation Environments (Domain 8).

Implements frontier manipulation capabilities:
1. Mid-trajectory reactive perturbations (displacement and impulse mid-approach)
2. Non-prehensile pushing followed by prehensile grasp-and-place
3. Multi-displacement stress tests and action-chunking phase probes
4. Multi-axis twist-lock and detent mechanisms (Family C, Tasks 227-233): articulated fixtures on native slide and
   hinge joints with spring stiffness, coupled only through contact (`assets/twist_lock_fixtures.xml`)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import numpy as np

from afford_pro.envs.parametric_env import (
    ParametricFrankaEnv,
    REST_LIN_SPEED,
    REST_ANG_SPEED,
)


class FrontierComplexEnv(ParametricFrankaEnv):
    """Frontier manipulation environment with reactive perturbations and non-prehensile phases."""

    def __init__(
        self,
        scene_xml_name: str,
        object_name: str,
        joint_name: str,
        nominal_pos: list[float] | np.ndarray,
        target_pos: list[float] | np.ndarray,
        instruction: str,
        task_type: Literal["pick_place", "insert", "press_button", "slide", "reactive_perturbation", "push_and_grasp",
                           "twist_lock"] = "pick_place",
        tolerance_xy: float = 0.005,
        tolerance_z: float = 0.005,
        tilt_tol_deg: float = 10.0,
        min_lift: float = 0.02,
        perturbation_step: int | None = None,
        perturbation_offset: list[float] | np.ndarray | None = None,
        second_perturbation_step: int | None = None,
        second_perturbation_offset: list[float] | np.ndarray | None = None,
        min_push_distance: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            scene_xml_name=scene_xml_name,
            object_name=object_name,
            joint_name=joint_name,
            nominal_pos=nominal_pos,
            target_pos=target_pos,
            instruction=instruction,
            task_type="pick_place" if task_type in ("reactive_perturbation", "push_and_grasp", "twist_lock") else task_type,
            tolerance_xy=tolerance_xy,
            tolerance_z=tolerance_z,
            tilt_tol_deg=tilt_tol_deg,
            min_lift=min_lift,
            **kwargs,
        )
        self.frontier_task_type = task_type
        self.perturbation_step = perturbation_step
        self.perturbation_offset = np.array(perturbation_offset, dtype=np.float64) if perturbation_offset is not None else None
        self.second_perturbation_step = second_perturbation_step
        self.second_perturbation_offset = np.array(second_perturbation_offset, dtype=np.float64) if second_perturbation_offset is not None else None
        self.min_push_distance = min_push_distance
        self.pushed_distance = 0.0
        self.initial_obj_xy = None
        self.perturbed_applied = False
        self.second_perturbed_applied = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        super().reset_objects(options=options)
        if self.target_qposadr is not None:
            self.initial_obj_xy = self.data.xpos[self.target_id][:2].copy()
        self.pushed_distance = 0.0
        self.perturbed_applied = False
        self.second_perturbed_applied = False

    def _apply_displacement(self, offset: np.ndarray) -> None:
        """Displace the manipulated free body mid-trajectory and zero its velocity."""
        if self.target_qposadr is None:
            return
        current_pos = self.data.qpos[self.target_qposadr : self.target_qposadr + 3].copy()
        new_pos = current_pos + offset
        
        # Verify support height at new position to ensure it remains resting
        support = self._support_height(new_pos)
        if support is not None:
            new_pos[2] = support + 0.002
            
        self.data.qpos[self.target_qposadr : self.target_qposadr + 3] = new_pos
        dof = self.free_qpos_to_dof[self.target_qposadr]
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _after_step(self) -> None:
        super()._after_step()
        
        # Track push distance
        if self.target_qposadr is not None and self.initial_obj_xy is not None:
            curr_xy = self.data.xpos[self.target_id][:2]
            dist = float(np.linalg.norm(curr_xy - self.initial_obj_xy))
            self.pushed_distance = max(self.pushed_distance, dist)

        # Apply primary mid-trajectory perturbation
        if self.perturbation_step is not None and not self.perturbed_applied:
            if self.current_step >= self.perturbation_step:
                offset = self.perturbation_offset if self.perturbation_offset is not None else np.array([0.04, 0.0, 0.0])
                self._apply_displacement(offset)
                self.perturbed_applied = True

        # Apply secondary mid-trajectory perturbation (for double displacement tasks)
        if self.second_perturbation_step is not None and not self.second_perturbed_applied:
            if self.current_step >= self.second_perturbation_step:
                offset = self.second_perturbation_offset if self.second_perturbation_offset is not None else np.array([0.0, 0.04, 0.0])
                self._apply_displacement(offset)
                self.second_perturbed_applied = True

    def check_success(self) -> bool:
        base_success = super().check_success()
        if not base_success:
            return False
            
        # For non-prehensile push-and-grasp tasks, verify that the push requirement was met
        if self.frontier_task_type == "push_and_grasp" and self.min_push_distance > 0.0:
            if self.pushed_distance < self.min_push_distance:
                return False

        return True


# ----------------------------------------------------------------- Family C: twist-lock and detent mechanisms
TWIST_ANGLE_TOL = np.radians(3.0)  # epsilon_q: terminal mechanism angle tolerance
TWIST_PRESS_TOL = 0.0015           # epsilon_z: terminal axial slide tolerance (m)
TWIST_HOLD_STEPS = 10              # the terminal state must hold for 0.5 s (20 Hz) with the hand released
LOW_FORCE_SLACK = np.radians(5.0)  # state-angle travel allowed while F_z < min_force (detent snap, settling)
CAP_THRESHOLD_FORCE = 20.0         # N: k_c = 2000 N/m times the 10 mm tooth gap


@dataclass(frozen=True)
class TwistLockSpec:
    """One articulated mechanism of `twist_lock_fixtures.xml` and its terminal configuration(s)."""

    kind: Literal["cap", "bayonet", "key"]
    body: str                          # body the robot grasps and turns (cap, cartridge, key)
    angle_joint: str                   # hinge whose angle is the mechanism state
    press_joint: str | None            # axial spring slide of the turned body (None for the key)
    stages: tuple[tuple[float, float | None], ...]  # terminal (angle rad, slide position m or None), in order
    start_angle: float = 0.0
    start_press: float = 0.0
    min_force: float = 0.0             # N; the state angle may only advance while the slide spring force is >= this
    min_insertion: float = 0.0         # m; key blade depth below the keyway top (key tasks)
    min_latch_retraction: float = 0.0  # m; latch slide displacement out of the bolt path (key tasks)
    min_bolt_travel: float = 0.0       # m; bolt slide opening (Task 231)


# Angles are about +z (counter-clockwise seen from above): the bayonets lock clockwise, the cap opens counter-clockwise.
TWIST_LOCK_MECHANISMS: dict[str, TwistLockSpec] = {
    "bayonet_lock": TwistLockSpec("bayonet", "bayonet_cartridge", "bayonet_twist", "bayonet_press",
                                  stages=((np.radians(-45.0), -0.008),)),
    "bayonet_unlock": TwistLockSpec("bayonet", "bayonet_cartridge", "bayonet_twist", "bayonet_press",
                                    stages=((0.0, 0.0),), start_angle=np.radians(-45.0), start_press=-0.008),
    "bayonet_two_step": TwistLockSpec("bayonet", "bayonet2_cartridge", "bayonet2_twist", "bayonet2_press",
                                      stages=((np.radians(-22.5), -0.008), (np.radians(-45.0), -0.008))),
    "key_turn": TwistLockSpec("key", "lock_key", "lock_plug_hinge", None, stages=((np.pi / 2.0, None),),
                              min_insertion=0.018, min_latch_retraction=0.004),
    "key_bolt": TwistLockSpec("key", "lock_key", "lock_plug_hinge", None, stages=((np.pi / 2.0, None),),
                              min_insertion=0.018, min_latch_retraction=0.004, min_bolt_travel=0.030),
    "cap_open": TwistLockSpec("cap", "safety_cap", "cap_ring_hinge", "cap_press", stages=((np.pi, None),),
                              min_force=CAP_THRESHOLD_FORCE),
    "cap_close": TwistLockSpec("cap", "safety_cap", "cap_ring_hinge", "cap_press", stages=((0.0, None),),
                               start_angle=np.pi, min_force=CAP_THRESHOLD_FORCE),
}


class TwistLockEnv(FrontierComplexEnv):
    """Family C: push-and-turn, bayonet, and key mechanisms built from slide and hinge joints with springs, coupled
    to each other only through contact (no welds, snapping, or teleporting after reset).

    Success Phi_C: every terminal stage of the mechanism is reached (state angle within 3 deg, axial slide within
    1.5 mm where specified) and held for 0.5 s with no finger touching the turned body (or the bolt); for the cap,
    the ring may advance by at most 5 deg in total while the cap spring force F_z = k_c dz (read from the
    slide-joint spring force) is below 20 N. Policies observe only RGB and proprioception.
    """

    def __init__(self, *args, mechanism: str, **kwargs):
        self.mechanism = mechanism
        self.mech = TWIST_LOCK_MECHANISMS[mechanism]
        super().__init__(*args, **kwargs)

    def _setup_task(self) -> None:
        super()._setup_task()
        m = self.model
        jid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)  # noqa: E731
        self.mech_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.mech.body)
        self.angle_qadr = int(m.jnt_qposadr[jid(self.mech.angle_joint)])
        self.press_qadr = self.press_dof = None
        if self.mech.press_joint is not None:
            self.press_qadr = int(m.jnt_qposadr[jid(self.mech.press_joint)])
            self.press_dof = int(m.jnt_dofadr[jid(self.mech.press_joint)])
        self.contact_body_ids = {self.mech_body_id}
        if self.mech.kind == "key":
            self.latch_qadr = int(m.jnt_qposadr[jid("lock_latch_slide")])
            self.bolt_qadr = int(m.jnt_qposadr[jid("lock_bolt_slide")])
            self.contact_body_ids.add(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lock_bolt"))
            top = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lock_top_xp")
            mujoco.mj_kinematics(m, self.data)
            self.keyway_top_z = float(self.data.geom_xpos[top][2] + m.geom_size[top][2])
        # The cap spins on its own hinge (tooth phase relative to the ring); other mechanisms have no such joint.
        self.spin_qadr = int(m.jnt_qposadr[jid("cap_spin")]) if self.mech.kind == "cap" else None
        self._reset_tracking()

    # ----------------------------------------------------------------- State
    @property
    def mechanism_angle(self) -> float:
        return float(self.data.qpos[self.angle_qadr])

    @property
    def press_position(self) -> float:
        """Axial slide position (m, negative when compressed); 0 for mechanisms without one."""
        return 0.0 if self.press_qadr is None else float(self.data.qpos[self.press_qadr])

    @property
    def axial_force(self) -> float:
        """F_z (N): the slide-joint spring force, k (springref - z), positive when the spring is compressed."""
        return 0.0 if self.press_dof is None else float(self.data.qfrc_spring[self.press_dof])

    @property
    def insertion_depth(self) -> float:
        """Key blade tip depth below the keyway top (m); the key body origin is the blade tip."""
        return self.keyway_top_z - float(self.data.xpos[self.mech_body_id][2]) if self.mech.kind == "key" else 0.0

    def hand_released(self) -> bool:
        return not any(self.body_contacts(b) & self.finger_body_ids for b in self.contact_body_ids)

    def stage_reached(self, stage: int) -> bool:
        angle, press = self.mech.stages[stage]
        if abs(self.mechanism_angle - angle) > TWIST_ANGLE_TOL:
            return False
        if press is not None and abs(self.press_position - press) > TWIST_PRESS_TOL:
            return False
        if self.mech.kind == "key":
            if self.insertion_depth < self.mech.min_insertion:
                return False
            if float(self.data.qpos[self.latch_qadr]) < self.mech.min_latch_retraction:
                return False
            if float(self.data.qpos[self.bolt_qadr]) < self.mech.min_bolt_travel:
                return False
        return True

    # ----------------------------------------------------------------- Reset, tracking, success
    def _reset_tracking(self) -> None:
        self.stage = 0
        self.stage_hold = 0
        self.twist_success = False
        self.low_force_travel = 0.0   # rad of state-angle travel while F_z < min_force
        self.forced_travel = 0.0      # rad of state-angle travel while F_z >= min_force
        self.peak_axial_force = 0.0
        self._last_angle = float(self.data.qpos[self.angle_qadr]) if hasattr(self, "angle_qadr") else 0.0

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        engine = self.perturbation_engine
        engine.apply_visual_perturbation(self.model)
        self.data.qpos[self.angle_qadr] = self.mech.start_angle
        if self.press_qadr is not None:
            self.data.qpos[self.press_qadr] = self.mech.start_press
        if self.mech.kind == "cap":  # the free-spinning cap's tooth phase relative to the ring
            self.data.qpos[self.spin_qadr] = engine.uniform(-np.pi, np.pi)
        elif self.mech.kind == "bayonet" and self.mech.start_press == 0.0:  # play of the pin in the start channel
            self.data.qpos[self.angle_qadr] += np.radians(engine.uniform(-0.5, 0.5))
        self._settle()
        self.dwell = 0
        self.start_fallback = False
        self.pushed_distance = 0.0
        self.perturbed_applied = self.second_perturbed_applied = False
        self._reset_tracking()

    def _hold_bodies(self) -> None:
        # Called before every physics substep (no constraint is applied): accumulate the state-angle travel
        # by the axial spring force acting at that substep.
        if not hasattr(self, "mech") or self.press_dof is None or self.mech.min_force <= 0.0:
            return
        angle = float(self.data.qpos[self.angle_qadr])
        travel = abs(angle - self._last_angle)
        self._last_angle = angle
        force = self.axial_force
        self.peak_axial_force = max(self.peak_axial_force, force)
        if force < self.mech.min_force:
            self.low_force_travel += travel
        else:
            self.forced_travel += travel

    def force_requirement_met(self) -> bool:
        return self.mech.min_force <= 0.0 or self.low_force_travel <= LOW_FORCE_SLACK

    def _after_step(self) -> None:
        super()._after_step()
        if self.twist_success:
            return
        if self.hand_released() and self.stage_reached(self.stage) and self.force_requirement_met():
            self.stage_hold += 1
        else:
            self.stage_hold = 0
        if self.stage_hold >= TWIST_HOLD_STEPS:
            if self.stage == len(self.mech.stages) - 1:
                self.twist_success = True
            else:
                self.stage += 1
                self.stage_hold = 0

    def check_success(self) -> bool:
        return bool(self.twist_success and self.force_requirement_met())
