"""Franka Industrial Shaft Assembly Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "industrial_assembly.xml"


class FrankaShaftAssemblyEnv(BaseFrankaEnv):
    """Task: Pick the machined transmission shaft from the tray and seat it into the jig bore."""

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
            "Pick the machined transmission shaft from the inbound tray and seat it into the central assembly jig bore."
        )

    def _setup_task(self) -> None:
        """Initialize task-specific IDs and nominal coordinates before reset."""
        # Object body IDs and joint addresses
        self.shaft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "machined_shaft")
        self.jig_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "assembly_jig")
        self.shaft_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_joint")
        ]

        # Nominal positions
        self.nominal_shaft_pos = np.array([0.40, 0.24, 0.050], dtype=np.float64)
        self.jig_bore_pos = np.array([0.55, 0.00, 0.116], dtype=np.float64)
        self.grasped_object = None
        self.seated = False

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset shaft and apply LIBERO-PRO style perturbations."""
        # 1. Spatial perturbation
        pos, quat = self.perturbation_engine.apply_spatial_perturbation(
            self.nominal_shaft_pos,
            jitter_xy=0.04,
            jitter_yaw_deg=30.0,
        )
        self.data.qpos[self.shaft_qposadr : self.shaft_qposadr + 3] = pos
        self.data.qpos[self.shaft_qposadr + 3 : self.shaft_qposadr + 7] = quat

        # 2. Visual / Lighting perturbation
        self.perturbation_engine.apply_visual_perturbation(self.model)

        # 3. Object material/color perturbation
        self.perturbation_engine.apply_object_perturbation(self.model, "shaft_core")
        self.grasped_object = None
        self.seated = False

    def _hold_bodies(self) -> None:
        """Keep a grasped body at the fingertips and a seated body at its goal."""
        if self.grasped_object == "shaft":
            self.pin_free_body(self.shaft_qposadr, self.tcp_position + np.array([0.0, 0.0, -0.015]))
        elif self.seated:
            self.pin_free_body(self.shaft_qposadr, self.jig_bore_pos)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # Track grasp state based on gripper proximity
        tip = self.tcp_position
        shaft_pos = self.data.xpos[self.shaft_id]
        dist = np.linalg.norm(tip - shaft_pos)

        # Check grasp closing
        if self.gripper_width <= 0.022 and dist < 0.06:
            self.grasped_object = "shaft"
            self.seated = False
        elif self.gripper_width >= 0.032:
            if self.grasped_object == "shaft":
                # Check if near jig bore on release: snap-in seating
                if np.linalg.norm(shaft_pos[:2] - self.jig_bore_pos[:2]) < 0.06:
                    self.seated = True
            self.grasped_object = None

        return super().step(action)

    def check_success(self) -> bool:
        """Success: shaft is seated inside bore and gripper has released."""
        shaft_pos = self.data.xpos[self.shaft_id]
        xy_err = np.linalg.norm(shaft_pos[:2] - self.jig_bore_pos[:2])
        z_err = abs(shaft_pos[2] - self.jig_bore_pos[2])
        is_seated = (xy_err < 0.03) and (z_err < 0.02)
        gripper_released = self.gripper_width >= 0.030
        return bool(is_seated and gripper_released)
