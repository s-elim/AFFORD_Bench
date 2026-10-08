"""Parametric Franka Environment for Rapid, Robust Task Definition across AFFORD-PRO-150."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


class ParametricFrankaEnv(BaseFrankaEnv):
    """Universal parameterized Franka manipulation environment supporting arbitrary tasks."""

    def __init__(
        self,
        scene_xml_name: str,
        object_name: str,
        joint_name: str,
        nominal_pos: list[float] | np.ndarray,
        target_pos: list[float] | np.ndarray,
        instruction: str,
        geom_name: str | None = None,
        task_type: Literal["pick_place", "insert", "press_button", "slide"] = "pick_place",
        tolerance_xy: float = 0.040,
        tolerance_z: float = 0.030,
        min_lift: float = 0.0,
        init_tcp_pos: list[float] | np.ndarray | None = None,
        perturbation_mode: str = "nominal",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 500,
        **kwargs,
    ):
        self.scene_xml_path = ASSET_DIR / scene_xml_name
        self.object_name = object_name
        self.joint_name = joint_name
        self.nominal_obj_pos = np.array(nominal_pos, dtype=np.float64)
        self.target_goal_pos = np.array(target_pos, dtype=np.float64)
        self.geom_name = geom_name or object_name
        self.task_type = task_type
        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z
        self.min_lift = min_lift
        self.init_tcp_pos = None if init_tcp_pos is None else np.array(init_tcp_pos, dtype=np.float64)
        self.perturbation_engine = PerturbationEngine(PerturbationConfig(mode=perturbation_mode))

        super().__init__(
            xml_path=self.scene_xml_path,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.language_instruction = instruction

    def _setup_task(self) -> None:
        """Cache MuJoCo addresses."""
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name)
        self.target_qposadr = self.model.jnt_qposadr[joint_id] if joint_id != -1 else None
        self.grasped_object = None
        self.seated = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset object and apply perturbations."""
        if self.target_qposadr is not None:
            pos, quat = self.perturbation_engine.apply_spatial_perturbation(
                self.nominal_obj_pos,
                jitter_xy=0.030,
                jitter_yaw_deg=45.0,
            )
            self.data.qpos[self.target_qposadr : self.target_qposadr + 3] = pos
            self.data.qpos[self.target_qposadr + 3 : self.target_qposadr + 7] = quat

        # Visual & object perturbations
        self.perturbation_engine.apply_visual_perturbation(self.model)
        self.perturbation_engine.apply_object_perturbation(self.model, self.geom_name)
        self.grasped_object = None
        self.seated = False

    def _settle(self) -> None:
        super()._settle()
        self.start_z = float(self.data.xpos[self.target_id][2])
        self.peak_z = self.start_z

    def _hold_bodies(self) -> None:
        """Keep a grasped body at the fingertips and a seated body at its goal."""
        if self.target_qposadr is None:
            return
        if self.grasped_object == self.object_name:
            self.pin_free_body(self.target_qposadr, self.tcp_position + np.array([0.0, 0.0, -0.010]))
        elif self.seated:
            self.pin_free_body(self.target_qposadr, self.target_goal_pos)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position

        if self.task_type in ("pick_place", "insert"):
            obj_pos = self.data.xpos[self.target_id]
            dist = np.linalg.norm(tip - obj_pos)

            gripper_closed = action[6] < 0
            if gripper_closed and dist < 0.065:
                self.grasped_object = self.object_name
                self.seated = False
            elif not gripper_closed:
                if self.grasped_object == self.object_name:
                    # Release near the goal seats the body there (kinematic snap).
                    if np.linalg.norm(obj_pos[:2] - self.target_goal_pos[:2]) < (self.tolerance_xy + 0.02):
                        self.seated = self.lift_satisfied()
                self.grasped_object = None

        out = super().step(action)
        self.peak_z = max(self.peak_z, float(self.data.xpos[self.target_id][2]))
        return out

    def lift_satisfied(self) -> bool:
        """True once the body has been raised at least `min_lift` above its settled start height."""
        return self.peak_z - self.start_z >= self.min_lift - 1e-6

    def check_success(self) -> bool:
        if self.task_type in ("pick_place", "insert"):
            obj_pos = self.data.xpos[self.target_id]
            xy_err = np.linalg.norm(obj_pos[:2] - self.target_goal_pos[:2])
            z_err = abs(obj_pos[2] - self.target_goal_pos[2])
            is_placed = (xy_err < self.tolerance_xy) and (z_err < self.tolerance_z)
            gripper_released = self.gripper_width >= 0.030
            return bool(is_placed and gripper_released and self.lift_satisfied())
        elif self.task_type == "press_button":
            # Success if TCP came within contact distance of button
            tip = self.tcp_position
            dist = np.linalg.norm(tip - self.target_goal_pos)
            return bool(dist < 0.025)
        return False
