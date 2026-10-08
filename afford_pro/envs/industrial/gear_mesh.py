"""Franka Industrial Gear Meshing Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "industrial_assembly.xml"


class FrankaGearMeshEnv(BaseFrankaEnv):
    """Task: Pick the bronze spur gear from the feeder spindle and mesh it onto the central transmission shaft."""

    def __init__(
        self,
        perturbation_mode: str = "nominal",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 500,
        **kwargs,
    ):
        self.perturbation_engine = PerturbationEngine(PerturbationConfig(mode=perturbation_mode))
        super().__init__(
            xml_path=SCENE_XML,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.language_instruction = (
            "Pick the bronze spur gear from the feeder spindle and mesh it onto the transmission shaft in the jig."
        )

    def _setup_task(self) -> None:
        """Initialize task-specific IDs and nominal coordinates."""
        self.gear_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "spur_gear")
        self.shaft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "machined_shaft")
        self.jig_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "assembly_jig")

        self.gear_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gear_joint")
        ]
        self.shaft_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_joint")
        ]

        # Nominal positions
        self.nominal_gear_pos = np.array([0.56, 0.18, 0.055], dtype=np.float64)
        self.jig_shaft_pos = np.array([0.55, 0.00, 0.116], dtype=np.float64)
        self.target_mesh_pos = np.array([0.55, 0.00, 0.090], dtype=np.float64)
        self.static_body_ids.add(self.shaft_id)
        self.grasped_object = None
        self.seated = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset shaft (seated in jig) and gear (on feeder with perturbations)."""
        # Shaft starts already seated in jig bore
        self.data.qpos[self.shaft_qposadr : self.shaft_qposadr + 3] = self.jig_shaft_pos
        self.data.qpos[self.shaft_qposadr + 3 : self.shaft_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 1. Spatial perturbation on gear
        pos, quat = self.perturbation_engine.apply_spatial_perturbation(
            self.nominal_gear_pos,
            jitter_xy=0.035,
            jitter_yaw_deg=45.0,
        )
        self.data.qpos[self.gear_qposadr : self.gear_qposadr + 3] = pos
        self.data.qpos[self.gear_qposadr + 3 : self.gear_qposadr + 7] = quat

        # 2. Visual / Lighting perturbation
        self.perturbation_engine.apply_visual_perturbation(self.model)

        # 3. Object material/color perturbation
        self.perturbation_engine.apply_object_perturbation(self.model, "gear_disk")
        self.grasped_object = None
        self.seated = False

    def _hold_bodies(self) -> None:
        """Keep the shaft seated in the jig, a grasped gear at the fingertips, and a seated gear at its goal."""
        self.pin_free_body(self.shaft_qposadr, self.jig_shaft_pos)
        if self.grasped_object == "gear":
            self.pin_free_body(self.gear_qposadr, self.tcp_position + np.array([0.0, 0.0, -0.012]))
        elif self.seated:
            self.pin_free_body(self.gear_qposadr, self.target_mesh_pos)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position
        gear_pos = self.data.xpos[self.gear_id]
        dist = np.linalg.norm(tip - gear_pos)

        # Check grasp closing
        if self.gripper_width <= 0.024 and dist < 0.065:
            self.grasped_object = "gear"
            self.seated = False
        elif self.gripper_width >= 0.032:
            if self.grasped_object == "gear":
                # Check if near target mesh position on shaft
                if np.linalg.norm(gear_pos[:2] - self.target_mesh_pos[:2]) < 0.05:
                    self.seated = True
            self.grasped_object = None

        return super().step(action)

    def check_success(self) -> bool:
        """Success: gear is seated around shaft and gripper has released."""
        gear_pos = self.data.xpos[self.gear_id]
        xy_err = np.linalg.norm(gear_pos[:2] - self.target_mesh_pos[:2])
        z_err = abs(gear_pos[2] - self.target_mesh_pos[2])
        is_meshed = (xy_err < 0.035) and (z_err < 0.025)
        gripper_released = self.gripper_width >= 0.030
        return bool(is_meshed and gripper_released)
