"""Frontier Complex Manipulation Environments (Domain 8).

Implements frontier manipulation capabilities:
1. Mid-trajectory reactive perturbations (displacement and impulse mid-approach)
2. Non-prehensile pushing followed by prehensile grasp-and-place
3. Multi-displacement stress tests and action-chunking phase probes
"""
from __future__ import annotations

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
        task_type: Literal["pick_place", "insert", "press_button", "slide", "reactive_perturbation", "push_and_grasp"] = "pick_place",
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
            task_type="pick_place" if task_type in ("reactive_perturbation", "push_and_grasp") else task_type,
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
