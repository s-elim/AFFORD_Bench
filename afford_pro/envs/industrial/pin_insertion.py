"""Franka Industrial Dowel Pin Insertion Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "industrial_assembly.xml"


class FrankaPinInsertionEnv(BaseFrankaEnv):
    """Task: Pick the hardened dowel pin from the cup feeder and insert it into the fixture locating hole."""

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
            "Pick the hardened steel dowel pin from the feeder cup and insert it into the locating hole on the assembly jig."
        )

    def _setup_task(self) -> None:
        """Initialize task-specific IDs and nominal coordinates."""
        self.pin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "dowel_pin")
        self.jig_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "assembly_jig")
        self.pin_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pin_joint")
        ]

        # Nominal positions
        self.nominal_pin_pos = np.array([0.42, 0.10, 0.035], dtype=np.float64)
        self.target_hole_pos = np.array([0.55, 0.055, 0.080], dtype=np.float64)
        self.grasped_object = None
        self.seated = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset pin with LIBERO-PRO style perturbations."""
        # 1. Spatial perturbation
        pos, quat = self.perturbation_engine.apply_spatial_perturbation(
            self.nominal_pin_pos,
            jitter_xy=0.025,
            jitter_yaw_deg=30.0,
        )
        self.data.qpos[self.pin_qposadr : self.pin_qposadr + 3] = pos
        self.data.qpos[self.pin_qposadr + 3 : self.pin_qposadr + 7] = quat

        # 2. Visual / Lighting perturbation
        self.perturbation_engine.apply_visual_perturbation(self.model)

        # 3. Object material/color perturbation
        self.perturbation_engine.apply_object_perturbation(self.model, "pin_core")
        self.grasped_object = None
        self.seated = False

    def _hold_bodies(self) -> None:
        """Keep a grasped body at the fingertips and a seated body at its goal."""
        if self.grasped_object == "pin":
            self.pin_free_body(self.pin_qposadr, self.tcp_position + np.array([0.0, 0.0, -0.010]))
        elif self.seated:
            self.pin_free_body(self.pin_qposadr, self.target_hole_pos)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position
        pin_pos = self.data.xpos[self.pin_id]
        dist = np.linalg.norm(tip - pin_pos)

        # Check grasp closing
        if self.gripper_width <= 0.020 and dist < 0.055:
            self.grasped_object = "pin"
            self.seated = False
        elif self.gripper_width >= 0.030:
            if self.grasped_object == "pin":
                # Check if near locating hole on release: snap-in insertion
                if np.linalg.norm(pin_pos[:2] - self.target_hole_pos[:2]) < 0.045:
                    self.seated = True
            self.grasped_object = None

        return super().step(action)

    def check_success(self) -> bool:
        """Success: pin is inserted in jig locating hole and gripper released."""
        pin_pos = self.data.xpos[self.pin_id]
        xy_err = np.linalg.norm(pin_pos[:2] - self.target_hole_pos[:2])
        z_err = abs(pin_pos[2] - self.target_hole_pos[2])
        is_inserted = (xy_err < 0.030) and (z_err < 0.025)
        gripper_released = self.gripper_width >= 0.028
        return bool(is_inserted and gripper_released)
