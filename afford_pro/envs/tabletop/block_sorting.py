"""Franka Tabletop Color & Geometry Block Sorting Environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
SCENE_XML = ASSET_DIR / "tabletop_sorting.xml"


class FrankaBlockSortingEnv(BaseFrankaEnv):
    """Task: Pick specified colored object and place it into the matching sorting bin."""

    def __init__(
        self,
        target_color: Literal["red", "green", "blue"] = "red",
        perturbation_mode: str = "nominal",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 500,
        **kwargs,
    ):
        self.target_color = target_color
        self.perturbation_engine = PerturbationEngine(PerturbationConfig(mode=perturbation_mode))
        super().__init__(
            xml_path=SCENE_XML,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        obj_desc = {
            "red": "red cube block",
            "green": "green cylinder block",
            "blue": "blue cube block",
        }[target_color]
        self.language_instruction = (
            f"Pick the {obj_desc} and place it into the {target_color} sorting bin."
        )

    def _setup_task(self) -> None:
        """Initialize object and bin IDs."""
        self.obj_ids = {
            "red": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"),
            "green": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "green_block"),
            "blue": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_block"),
        }
        self.bin_ids = {
            "red": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_bin"),
            "green": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "green_bin"),
            "blue": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_bin"),
        }
        self.qposadrs = {
            "red": self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_joint")],
            "green": self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "green_joint")],
            "blue": self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "blue_joint")],
        }

        self.nominal_positions = {
            "red": np.array([0.45, -0.08, 0.025], dtype=np.float64),
            "green": np.array([0.52, 0.07, 0.025], dtype=np.float64),
            "blue": np.array([0.38, 0.06, 0.025], dtype=np.float64),
        }
        self.bin_targets = {
            "red": np.array([0.40, -0.24, 0.035], dtype=np.float64),
            "green": np.array([0.58, 0.00, 0.035], dtype=np.float64),
            "blue": np.array([0.40, 0.24, 0.035], dtype=np.float64),
        }

        self.target_id = self.obj_ids[self.target_color]
        self.target_qposadr = self.qposadrs[self.target_color]
        self.target_bin_pos = self.bin_targets[self.target_color]
        self.grasped_object = None

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Reset all objects with perturbations applied to the target object."""
        for color, adr in self.qposadrs.items():
            base_pos = self.nominal_positions[color]
            if color == self.target_color:
                pos, quat = self.perturbation_engine.apply_spatial_perturbation(
                    base_pos,
                    jitter_xy=0.035,
                    jitter_yaw_deg=45.0,
                )
            else:
                pos = base_pos.copy()
                quat = np.array([1.0, 0.0, 0.0, 0.0])

            self.data.qpos[adr : adr + 3] = pos
            self.data.qpos[adr + 3 : adr + 7] = quat

        # Visual and object perturbations
        self.perturbation_engine.apply_visual_perturbation(self.model)
        self.perturbation_engine.apply_object_perturbation(self.model, f"{self.target_color}_geom")
        self.grasped_object = None

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        tip = self.tcp_position
        obj_pos = self.data.xpos[self.target_id]
        dist = np.linalg.norm(tip - obj_pos)

        gripper_closed = action[6] < 0
        if gripper_closed and dist < 0.060:
            self.grasped_object = self.target_color
        elif not gripper_closed:
            if self.grasped_object == self.target_color:
                if np.linalg.norm(obj_pos[:2] - self.target_bin_pos[:2]) < 0.055:
                    self.data.qpos[self.target_qposadr : self.target_qposadr + 3] = self.target_bin_pos
                    self.data.qpos[self.target_qposadr + 3 : self.target_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.grasped_object = None

        if self.grasped_object == self.target_color:
            self.data.qpos[self.target_qposadr : self.target_qposadr + 3] = tip + np.array([0.0, 0.0, -0.010])
            self.data.qpos[self.target_qposadr + 3 : self.target_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        return super().step(action)

    def check_success(self) -> bool:
        """Success: target object is inside destination bin and gripper released."""
        obj_pos = self.data.xpos[self.target_id]
        xy_err = np.linalg.norm(obj_pos[:2] - self.target_bin_pos[:2])
        z_err = abs(obj_pos[2] - self.target_bin_pos[2])
        is_sorted = (xy_err < 0.040) and (z_err < 0.030)
        gripper_released = self.gripper_width >= 0.030
        return bool(is_sorted and gripper_released)


class FrankaSortRedEnv(FrankaBlockSortingEnv):
    def __init__(self, **kwargs):
        super().__init__(target_color="red", **kwargs)


class FrankaSortGreenEnv(FrankaBlockSortingEnv):
    def __init__(self, **kwargs):
        super().__init__(target_color="green", **kwargs)


class FrankaSortBlueEnv(FrankaBlockSortingEnv):
    def __init__(self, **kwargs):
        super().__init__(target_color="blue", **kwargs)
