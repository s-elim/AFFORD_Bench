"""Hierarchical (L1-L5) perturbation engine for MuJoCo scenes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import mujoco
import numpy as np


PerturbationMode = Literal[
    "nominal",       # L1: in-distribution scatter
    "p1_object",     # object color shift
    "p2_spatial",    # L2: scatter outside the L1 support
    "p3_visual",     # L3: lighting
    "p4_physical",   # L4: friction and mass
    "combined_ood",  # L5: L2 + L3 + L4 + color
]
SPATIAL_MODES = ("p2_spatial", "combined_ood")
VISUAL_MODES = ("p3_visual", "combined_ood")
OBJECT_MODES = ("p1_object", "combined_ood")
PHYSICAL_MODES = ("p4_physical", "combined_ood")


@dataclass
class PerturbationConfig:
    mode: str = "nominal"
    seed: int | None = None
    # L1 nominal scatter of the manipulated body.
    nominal_xy_m: float = 0.07
    nominal_yaw_deg: float = 45.0
    nominal_tilt_deg: float = 10.0
    # L1 jitter of the initial TCP.
    tcp_jitter_xy_m: float = 0.015
    # L2 spatial shift: radial offset in [min, max] and yaw beyond the L1 range.
    spatial_radius_m: tuple[float, float] = (0.07, 0.12)
    spatial_yaw_deg: tuple[float, float] = (45.0, 90.0)
    # L3 visual.
    light_pos_jitter_m: float = 0.5
    light_intensity_range: tuple[float, float] = (0.5, 1.5)
    color_jitter_factor: float = 0.25
    # L4 physical: sliding friction of pads and object, and object mass scale.
    friction_range: tuple[float, float] = (0.12, 0.25)
    mass_scale_range: tuple[float, float] = (1.8, 2.5)


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), "XYZ")
    return quat


class PerturbationEngine:
    """Samples per-episode layouts and model perturbations; restores nominal model parameters first."""

    def __init__(self, config: PerturbationConfig | None = None):
        self.config = config or PerturbationConfig()
        self.mode = str(self.config.mode).lower()
        self.rng = np.random.default_rng(self.config.seed)
        self._nominal: dict[str, np.ndarray] | None = None

    def set_seed(self, seed: int):
        self.config.seed = seed
        self.rng = np.random.default_rng(seed)

    def begin_episode(self, model: mujoco.MjModel, seed: int) -> None:
        """Re-seed the sampler and restore nominal lights, colors, friction, and masses.

        Perturbations scale model parameters in place, so without the restore their severity would
        compound across episodes.
        """
        self.set_seed(seed)
        if self._nominal is None:
            self._nominal = {
                "light_pos": model.light_pos.copy(),
                "light_diffuse": model.light_diffuse.copy(),
                "geom_rgba": model.geom_rgba.copy(),
                "geom_friction": model.geom_friction.copy(),
                "body_mass": model.body_mass.copy(),
                "body_inertia": model.body_inertia.copy(),
            }
        model.light_pos[:] = self._nominal["light_pos"]
        model.light_diffuse[:] = self._nominal["light_diffuse"]
        model.geom_rgba[:] = self._nominal["geom_rgba"]
        model.geom_friction[:] = self._nominal["geom_friction"]
        model.body_mass[:] = self._nominal["body_mass"]
        model.body_inertia[:] = self._nominal["body_inertia"]

    def sample_pose(self, base_pos: np.ndarray, tilt: bool = True, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Initial pose of the manipulated body: L2 shift in spatial modes, L1 scatter otherwise, with the
        translation scaled by `scale` (the reset shrinks it when no valid start is found at full scale).
        Roll and pitch are applied only when `tilt` is set (bodies stable at twice the tilt bound)."""
        c = self.config
        pos = np.asarray(base_pos, dtype=np.float64).copy()
        if self.mode in SPATIAL_MODES:
            radius = self.rng.uniform(*c.spatial_radius_m) * scale
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            pos[:2] += radius * np.array([np.cos(angle), np.sin(angle)])
            yaw = np.radians(self.rng.uniform(*c.spatial_yaw_deg)) * self.rng.choice([-1.0, 1.0])
        else:
            pos[:2] += self.rng.uniform(-c.nominal_xy_m, c.nominal_xy_m, size=2) * scale
            yaw = np.radians(self.rng.uniform(-c.nominal_yaw_deg, c.nominal_yaw_deg))
        roll, pitch = np.radians(self.rng.uniform(-c.nominal_tilt_deg, c.nominal_tilt_deg, size=2)) * float(tilt)
        return pos, _quat_from_euler(roll, pitch, yaw)

    def jitter_point(self, point: np.ndarray) -> np.ndarray:
        """Apply the L1 (x, y) jitter to the initial TCP target."""
        out = np.asarray(point, dtype=np.float64).copy()
        out[:2] += self.rng.uniform(-self.config.tcp_jitter_xy_m, self.config.tcp_jitter_xy_m, size=2)
        return out

    def uniform(self, low, high, size=None):
        return self.rng.uniform(low, high, size=size)

    def apply_visual_perturbation(self, model: mujoco.MjModel):
        """Perturb lighting position, direction, and intensity in MuJoCo model."""
        if self.mode not in VISUAL_MODES:
            return
        for light_id in range(model.nlight):
            model.light_pos[light_id] += self.rng.uniform(-self.config.light_pos_jitter_m, self.config.light_pos_jitter_m, size=3)
            factor = self.rng.uniform(*self.config.light_intensity_range)
            tint = self.rng.uniform(0.9, 1.1, size=3)
            model.light_diffuse[light_id] = np.clip(model.light_diffuse[light_id] * factor * tint, 0.1, 1.0)

    def apply_object_perturbation(self, model: mujoco.MjModel, geom_ids: list[int]):
        """Shift the color of the manipulated body's geoms."""
        if self.mode not in OBJECT_MODES:
            return
        jitter = self.rng.uniform(1.0 - self.config.color_jitter_factor, 1.0 + self.config.color_jitter_factor, size=3)
        for g in geom_ids:
            model.geom_rgba[g][:3] = np.clip(model.geom_rgba[g][:3] * jitter, 0.0, 1.0)

    def apply_physical_perturbation(self, model: mujoco.MjModel, body_id: int, object_geom_ids: list[int],
                                    pad_geom_ids: list[int]) -> None:
        """L4: slippery pads and object (sliding friction in friction_range) and a heavier object.

        MuJoCo combines contact friction by the maximum of the two geoms, so pads and object are
        both set; mass and rotational inertia scale together, keeping the center of mass.
        """
        if self.mode not in PHYSICAL_MODES:
            return
        mu = self.rng.uniform(*self.config.friction_range)
        for g in list(object_geom_ids) + list(pad_geom_ids):
            model.geom_friction[g][0] = mu
        scale = self.rng.uniform(*self.config.mass_scale_range)
        model.body_mass[body_id] *= scale
        model.body_inertia[body_id] *= scale
