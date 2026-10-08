"""Franka Culinary Burger Patty Stacking Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "kitchen_burger.xml"


class FrankaPattyStackEnv(BaseFrankaEnv):
    """Task: Pick the grilled patty from the grill station and place it onto the bottom bun."""

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
            "Pick the grilled beef burger patty from the cast-iron grill and place it onto the bottom bun on the assembly board."
        )

    def _setup_task(self) -> None:
        """Initialize task-specific IDs and nominal coordinates."""
        self.patty_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "burger_patty")
        self.bun_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bottom_bun")
        self.board_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "assembly_board")

        self.patty_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "patty_joint")
        ]
        self.bun_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "b_bun_joint")
        ]

        # Nominal positions
        self.nominal_patty_pos = np.array([0.58, 0.20, 0.042], dtype=np.float64)
        self.nominal_bun_pos = np.array([0.52, 0.00, 0.026], dtype=np.float64)
        self.target_stack_pos = np.array([0.52, 0.00, 0.042], dtype=np.float64)
        self.grasped_object = None
        self.seated = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset bottom bun (on board) and patty (on grill with perturbations)."""
        # Bottom bun starts on assembly board
        self.data.qpos[self.bun_qposadr : self.bun_qposadr + 3] = self.nominal_bun_pos
        self.data.qpos[self.bun_qposadr + 3 : self.bun_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 1. Spatial perturbation on patty
        pos, quat = self.perturbation_engine.apply_spatial_perturbation(
            self.nominal_patty_pos,
            jitter_xy=0.030,
            jitter_yaw_deg=45.0,
        )
        self.data.qpos[self.patty_qposadr : self.patty_qposadr + 3] = pos
        self.data.qpos[self.patty_qposadr + 3 : self.patty_qposadr + 7] = quat

        # 2. Visual / Lighting perturbation
        self.perturbation_engine.apply_visual_perturbation(self.model)

        # 3. Object material/color perturbation
        self.perturbation_engine.apply_object_perturbation(self.model, "patty_core")
        self.grasped_object = None
        self.seated = False

    def _hold_bodies(self) -> None:
        """Keep a grasped body at the fingertips and a seated body at its goal."""
        if self.grasped_object == "patty":
            self.pin_free_body(self.patty_qposadr, self.tcp_position + np.array([0.0, 0.0, -0.010]))
        elif self.seated:
            self.pin_free_body(self.patty_qposadr, self.target_stack_pos)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position
        patty_pos = self.data.xpos[self.patty_id]
        dist = np.linalg.norm(tip - patty_pos)

        # Check grasp closing
        gripper_closed = action[6] < 0
        if gripper_closed and dist < 0.065:
            self.grasped_object = "patty"
            self.seated = False
        elif not gripper_closed:
            if self.grasped_object == "patty":
                # Check if near target stack position on bun
                if np.linalg.norm(patty_pos[:2] - self.target_stack_pos[:2]) < 0.055:
                    self.seated = True
            self.grasped_object = None

        return super().step(action)

    def check_success(self) -> bool:
        """Success: patty is resting on bottom bun and gripper has opened."""
        patty_pos = self.data.xpos[self.patty_id]
        xy_err = np.linalg.norm(patty_pos[:2] - self.target_stack_pos[:2])
        z_err = abs(patty_pos[2] - self.target_stack_pos[2])
        is_stacked = (xy_err < 0.035) and (z_err < 0.025)
        gripper_released = self.gripper_width >= 0.030
        return bool(is_stacked and gripper_released)
