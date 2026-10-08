"""Franka Culinary Coffee Mug to Espresso Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "kitchen_coffee.xml"


class FrankaCoffeeMugEnv(BaseFrankaEnv):
    """Task: Pick the ceramic coffee mug from the storage bin and place it onto the espresso drip tray."""

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
            "Pick the ceramic coffee mug from the storage bin and place it onto the espresso machine drip tray."
        )

    def _setup_task(self) -> None:
        """Initialize task-specific IDs and nominal coordinates."""
        self.mug_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_mug")
        self.machine_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
        self.mug_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
        ]

        # Nominal positions
        self.nominal_mug_pos = np.array([0.38, 0.22, 0.038], dtype=np.float64)
        self.target_drip_tray_pos = np.array([0.48, -0.15, 0.038], dtype=np.float64)
        self.grasped_object = None
        self.seated = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset mug with LIBERO-PRO style perturbations."""
        # 1. Spatial perturbation on mug
        pos, quat = self.perturbation_engine.apply_spatial_perturbation(
            self.nominal_mug_pos,
            jitter_xy=0.030,
            jitter_yaw_deg=45.0,
        )
        self.data.qpos[self.mug_qposadr : self.mug_qposadr + 3] = pos
        self.data.qpos[self.mug_qposadr + 3 : self.mug_qposadr + 7] = quat

        # 2. Visual / Lighting perturbation
        self.perturbation_engine.apply_visual_perturbation(self.model)

        # 3. Object material/color perturbation
        self.perturbation_engine.apply_object_perturbation(self.model, "mug_body")
        self.grasped_object = None
        self.seated = False

    def _hold_bodies(self) -> None:
        """Keep a grasped body at the fingertips and a seated body at its goal."""
        if self.grasped_object == "mug":
            self.pin_free_body(self.mug_qposadr, self.tcp_position + np.array([0.0, 0.0, -0.010]))
        elif self.seated:
            self.pin_free_body(self.mug_qposadr, self.target_drip_tray_pos)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position
        mug_pos = self.data.xpos[self.mug_id]
        dist = np.linalg.norm(tip - mug_pos)

        # Check grasp closing
        gripper_closed = action[6] < 0
        if gripper_closed and dist < 0.065:
            self.grasped_object = "mug"
            self.seated = False
        elif not gripper_closed:
            if self.grasped_object == "mug":
                if np.linalg.norm(mug_pos[:2] - self.target_drip_tray_pos[:2]) < 0.055:
                    self.seated = True
            self.grasped_object = None

        return super().step(action)

    def check_success(self) -> bool:
        """Success: mug is resting on drip tray and gripper released."""
        mug_pos = self.data.xpos[self.mug_id]
        xy_err = np.linalg.norm(mug_pos[:2] - self.target_drip_tray_pos[:2])
        z_err = abs(mug_pos[2] - self.target_drip_tray_pos[2])
        is_placed = (xy_err < 0.035) and (z_err < 0.025)
        gripper_released = self.gripper_width >= 0.030
        return bool(is_placed and gripper_released)
