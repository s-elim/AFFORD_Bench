"""LIBERO-PRO Style 5-Axis Systematic Perturbation Engine for MuJoCo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import mujoco
import numpy as np


PerturbationMode = Literal[
    "nominal",       # In-Distribution (ID) baseline
    "P1_object",     # Object visual/scale variations
    "P2_spatial",    # Spatial layout and position jitter
    "P3_visual",     # Lighting, shadows, and table appearance
    "P4_distractor", # Clutter and non-target distractors
    "P5_semantic",   # Language instruction paraphrases
    "combined_ood",  # Full combined multi-axis perturbation
]


@dataclass
class PerturbationConfig:
    mode: PerturbationMode = "nominal"
    seed: int | None = None
    spatial_jitter_xy_m: float = 0.08
    spatial_jitter_yaw_deg: float = 45.0
    light_pos_jitter_m: float = 0.5
    light_intensity_range: tuple[float, float] = (0.5, 1.5)
    color_jitter_factor: float = 0.25
    num_distractors: int = 2


class PerturbationEngine:
    """Systematically applies out-of-distribution perturbations to a MuJoCo model and data."""

    def __init__(self, config: PerturbationConfig | None = None):
        self.config = config or PerturbationConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def set_seed(self, seed: int):
        self.config.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply_spatial_perturbation(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray | None = None,
        jitter_xy: float | None = None,
        jitter_yaw_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random bounded (x, y) displacement and yaw rotation."""
        mode_str = str(self.config.mode).lower()
        if mode_str not in ("p2_spatial", "combined_ood"):
            quat = base_quat if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
            return base_pos.copy(), quat

        j_xy = jitter_xy or self.config.spatial_jitter_xy_m
        j_yaw = np.radians(jitter_yaw_deg or self.config.spatial_jitter_yaw_deg)

        dx = self.rng.uniform(-j_xy, j_xy)
        dy = self.rng.uniform(-j_xy, j_xy)
        perturbed_pos = base_pos.copy()
        perturbed_pos[0] += dx
        perturbed_pos[1] += dy

        # Random yaw perturbation
        d_yaw = self.rng.uniform(-j_yaw, j_yaw)
        # Quaternion for yaw rotation around Z: [cos(theta/2), 0, 0, sin(theta/2)]
        yaw_quat = np.array([np.cos(d_yaw / 2), 0.0, 0.0, np.sin(d_yaw / 2)], dtype=np.float64)

        if base_quat is not None:
            # Multiply quats
            out_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(out_quat, yaw_quat, base_quat)
        else:
            out_quat = yaw_quat

        return perturbed_pos, out_quat

    def apply_visual_perturbation(self, model: mujoco.MjModel):
        """Perturb lighting position, direction, and intensity in MuJoCo model."""
        mode_str = str(self.config.mode).lower()
        if mode_str not in ("p3_visual", "combined_ood"):
            return

        for light_id in range(model.nlight):
            # Jitter light position
            j_pos = self.rng.uniform(-self.config.light_pos_jitter_m, self.config.light_pos_jitter_m, size=3)
            model.light_pos[light_id] += j_pos

            # Jitter intensity / color temperature
            factor = self.rng.uniform(*self.config.light_intensity_range)
            tint = self.rng.uniform(0.9, 1.1, size=3)
            model.light_diffuse[light_id] = np.clip(model.light_diffuse[light_id] * factor * tint, 0.1, 1.0)

    def apply_object_perturbation(self, model: mujoco.MjModel, geom_name: str):
        """Apply color/material shift to object geometry."""
        mode_str = str(self.config.mode).lower()
        if mode_str not in ("p1_object", "combined_ood"):
            return

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id != -1:
            # Color jitter
            jitter = self.rng.uniform(
                1.0 - self.config.color_jitter_factor,
                1.0 + self.config.color_jitter_factor,
                size=3,
            )
            model.geom_rgba[geom_id][:3] = np.clip(model.geom_rgba[geom_id][:3] * jitter, 0.0, 1.0)
