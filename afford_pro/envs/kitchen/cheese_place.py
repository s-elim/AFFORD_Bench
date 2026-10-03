"""Franka Culinary Cheese Placement Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "kitchen_burger.xml"


class FrankaCheesePlaceEnv(BaseFrankaEnv):
    """Task: Pick the cheddar cheese slice from the topping tray and place it onto the patty."""

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
            "Pick the cheddar cheese slice from the chilled topping tray and place it onto the burger patty."
        )

    def _setup_task(self) -> None:
        """Initialize task-specific IDs and nominal coordinates."""
        self.cheese_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cheese_slice")
        self.patty_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "burger_patty")
        self.bun_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bottom_bun")

        self.cheese_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cheese_joint")
        ]
        self.patty_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "patty_joint")
        ]
        self.bun_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "b_bun_joint")
        ]

        # Nominal positions
        self.nominal_cheese_pos = np.array([0.58, -0.12, 0.026], dtype=np.float64)
        self.nominal_bun_pos = np.array([0.52, 0.00, 0.026], dtype=np.float64)
        self.nominal_patty_pos = np.array([0.52, 0.00, 0.042], dtype=np.float64)
        self.target_cheese_pos = np.array([0.52, 0.00, 0.052], dtype=np.float64)
        self.grasped_object = None

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset bottom bun and patty (already stacked on board) and cheese (with perturbations)."""
        self.data.qpos[self.bun_qposadr : self.bun_qposadr + 3] = self.nominal_bun_pos
        self.data.qpos[self.bun_qposadr + 3 : self.bun_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qpos[self.patty_qposadr : self.patty_qposadr + 3] = self.nominal_patty_pos
        self.data.qpos[self.patty_qposadr + 3 : self.patty_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 1. Spatial perturbation on cheese slice
        pos, quat = self.perturbation_engine.apply_spatial_perturbation(
            self.nominal_cheese_pos,
            jitter_xy=0.025,
            jitter_yaw_deg=45.0,
        )
        self.data.qpos[self.cheese_qposadr : self.cheese_qposadr + 3] = pos
        self.data.qpos[self.cheese_qposadr + 3 : self.cheese_qposadr + 7] = quat

        # 2. Visual / Lighting perturbation
        self.perturbation_engine.apply_visual_perturbation(self.model)

        # 3. Object material/color perturbation
        self.perturbation_engine.apply_object_perturbation(self.model, "cheese_body")
        self.grasped_object = None

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position
        cheese_pos = self.data.xpos[self.cheese_id]
        dist = np.linalg.norm(tip - cheese_pos)

        # Check grasp closing
        gripper_closed = action[6] < 0
        if gripper_closed and dist < 0.060:
            self.grasped_object = "cheese"
        elif not gripper_closed:
            if self.grasped_object == "cheese":
                if np.linalg.norm(cheese_pos[:2] - self.target_cheese_pos[:2]) < 0.055:
                    self.data.qpos[self.cheese_qposadr : self.cheese_qposadr + 3] = self.target_cheese_pos
                    self.data.qpos[self.cheese_qposadr + 3 : self.cheese_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.grasped_object = None

        if self.grasped_object == "cheese":
            self.data.qpos[self.cheese_qposadr : self.cheese_qposadr + 3] = tip + np.array([0.0, 0.0, -0.008])
            self.data.qpos[self.cheese_qposadr + 3 : self.cheese_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        return super().step(action)

    def check_success(self) -> bool:
        """Success: cheese is resting on patty and gripper has released."""
        cheese_pos = self.data.xpos[self.cheese_id]
        xy_err = np.linalg.norm(cheese_pos[:2] - self.target_cheese_pos[:2])
        z_err = abs(cheese_pos[2] - self.target_cheese_pos[2])
        is_placed = (xy_err < 0.035) and (z_err < 0.025)
        gripper_released = self.gripper_width >= 0.030
        return bool(is_placed and gripper_released)
