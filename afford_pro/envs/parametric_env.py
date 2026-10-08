"""Parametric Franka environment shared by all 210 AFFORD-Bench tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import GRIPPER_OPEN_WIDTH, BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig


ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

REST_LIN_SPEED = 0.02   # m/s: a placed body counts as at rest below this speed
REST_ANG_SPEED = 0.2    # rad/s
PRESS_DWELL_STEPS = 5   # consecutive 20 Hz steps the TCP must stay within tolerance (0.25 s)
PAD_BELOW_TCP = 0.0115  # finger pads extend this far below the TCP (m)
# (scatter scale, attempts) rounds of start rejection sampling before the nominal fallback.
RESET_SCATTER_SCHEDULE = ((1.0, 16), (0.5, 8), (0.25, 8), (0.1, 8))
WORKSPACE_X = (0.33, 0.66)
WORKSPACE_Y = (-0.32, 0.32)
CUBE_BODIES = ("red_block", "blue_block")
HOLD_HEIGHT = 0.05      # goals more than this above the support under them are hold goals (m)


class ParametricFrankaEnv(BaseFrankaEnv):
    """Pick-and-place, insertion, and reach-and-press tasks with physically verified success.

    Pick tasks succeed when the body rests (released, at rest, upright within `tilt_tol_deg`) within
    (tolerance_xy, tolerance_z) of the goal after having been lifted at least `min_lift`. The goal
    height is the rest height measured by dropping the body at the goal; goals where the body cannot
    rest (for example, positioning a pipette in mid-air) are hold goals, satisfied while the body is
    held in a two-finger grasp at the goal pose.
    """

    def __init__(
        self,
        scene_xml_name: str,
        object_name: str,
        joint_name: str,
        nominal_pos: list[float] | np.ndarray,
        target_pos: list[float] | np.ndarray,
        instruction: str,
        task_type: Literal["pick_place", "insert", "press_button", "slide"] = "pick_place",
        tolerance_xy: float = 0.008,
        tolerance_z: float = 0.008,
        tilt_tol_deg: float = 15.0,
        min_lift: float = 0.02,
        init_tcp_pos: list[float] | np.ndarray | None = None,
        scene_bodies: dict[str, list[float]] | None = None,
        distractors: bool = False,
        goal_ref_body: str | None = None,
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
        self.task_type = task_type
        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z
        self.tilt_tol = np.radians(tilt_tol_deg)
        self.min_lift = min_lift
        self.init_tcp_pos = None if init_tcp_pos is None else np.array(init_tcp_pos, dtype=np.float64)
        self.scene_bodies = scene_bodies or {}
        self.use_distractors = distractors
        self.goal_ref_body = goal_ref_body
        self.perturbation_engine = PerturbationEngine(PerturbationConfig(mode=perturbation_mode))
        self.goal_z = float(self.target_goal_pos[2])
        self.goal_xy = self.target_goal_pos[:2].copy()
        self.hold_goal = False

        super().__init__(
            xml_path=self.scene_xml_path,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        self.language_instruction = instruction
        self.press_goal = self._project_press_goal() if task_type == "press_button" else None
        if self.target_qposadr is not None:
            self._calibrate_goal()

    # ----------------------------------------------------------------- Setup
    def _setup_task(self) -> None:
        m = self.model
        self.target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name)
        is_free = joint_id != -1 and m.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        self.target_qposadr = int(m.jnt_qposadr[joint_id]) if is_free and self.task_type != "press_button" else None
        self.object_geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.target_id]
        self.pad_geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.finger_body_ids and m.geom_contype[g]]
        self.distractor_qposadr = [
            self.free_body_qposadr[b] for b in sorted(self.free_body_qposadr)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith("distractor")
        ]
        self.goal_ref_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.goal_ref_body) if self.goal_ref_body else -1
        self.scene_body_qposadr = {
            jn: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for jn in self.scene_bodies
        }
        self.start_z = 0.0
        self.peak_z = 0.0
        self.dwell = 0
        self.spawn_tilt_ok = self.target_qposadr is not None and self._tipping_angle() > 2.0 * np.radians(
            self.perturbation_engine.config.nominal_tilt_deg)

    def object_extent(self, direction: np.ndarray) -> tuple[float, float]:
        """Lowest and highest projection onto the unit world `direction` of the manipulated body's
        collision geoms (exact support functions for primitives, bounding boxes for meshes)."""
        m, d = self.model, self.data
        u = np.asarray(direction, dtype=np.float64)
        lo, hi = np.inf, -np.inf
        for g in self.object_geom_ids:
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            ul = rot.T @ u  # direction in the geom frame
            size, kind = m.geom_size[g], m.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                reach = float(np.abs(ul) @ size)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                reach = float(size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                reach = float(size[1] * abs(ul[2]) + size[0] * np.hypot(ul[0], ul[1]))
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                reach = float(size[1] * abs(ul[2]) + size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                reach = float(np.linalg.norm(size * ul))
            else:
                center, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                c = float(d.geom_xpos[g] @ u + center @ ul)
                lo, hi = min(lo, c - float(np.abs(ul) @ half)), max(hi, c + float(np.abs(ul) @ half))
                continue
            c = float(d.geom_xpos[g] @ u)
            lo, hi = min(lo, c - reach), max(hi, c + reach)
        return float(lo), float(hi)

    def object_extent_z(self) -> tuple[float, float]:
        """World z of the lowest and highest point of the manipulated body's collision geoms."""
        return self.object_extent(np.array([0.0, 0.0, 1.0]))

    def _tipping_angle(self) -> float:
        """Static tipping angle (rad) of the upright body: atan(base half-width / center-of-mass height)."""
        self.pin_free_body(self.target_qposadr, np.array([0.0, 0.0, 1.0]))
        mujoco.mj_forward(self.model, self.data)
        lo, _ = self.object_extent_z()
        m, d = self.model, self.data
        base = np.inf
        for g in self.object_geom_ids:
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            half = m.geom_aabb[g][3:]
            if d.geom_xpos[g][2] - half[2] <= lo + 0.003:
                base = min(base, min(half[0], half[1]) + np.linalg.norm(d.geom_xpos[g][:2] - d.xpos[self.target_id][:2]))
        com_height = d.xipos[self.target_id][2] - lo
        return float(np.arctan2(base, com_height))

    def tilt(self) -> float:
        """Angle (rad) between the body's up axis and world z; for cubes, any face may face up."""
        rot = self.data.xmat[self.target_id].reshape(3, 3)
        if self.object_name in CUBE_BODIES:
            return float(np.arccos(np.clip(np.abs(rot[2]).max(), -1.0, 1.0)))
        return float(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))

    def ray_vertical(self, start: np.ndarray, up: bool = False) -> float | None:
        """Height of the first colliding, non-robot surface straight below (or above) `start`, ignoring
        the manipulated body and passing through visual-only geoms."""
        m, d = self.model, self.data
        sign = 1.0 if up else -1.0
        geom_id = np.zeros(1, dtype=np.int32)
        point = np.asarray(start, dtype=np.float64).copy()
        for _ in range(8):
            dist = mujoco.mj_ray(m, d, point, np.array([0.0, 0.0, sign]), None, 1, self.target_id, geom_id)
            if dist < 0:
                return None
            g = int(geom_id[0])
            hit_z = point[2] + sign * dist
            if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body_rootid[m.geom_bodyid[g]] != self.robot_root_id:
                return float(hit_z)
            point = np.array([point[0], point[1], hit_z + sign * 1e-4])
        return None

    def graspable(self) -> bool:
        """True if a top-down parallel-jaw grasp of the body exists: at one of four yaws the hand, with the
        fingers 1 cm below the body top, is collision-free both opened up to 5 mm beyond the body and closed
        onto it. Starts without one (a body pressed against a compartment wall) cannot be solved by grasping."""
        if self.target_qposadr is None:
            return True
        lo, hi = self.object_extent_z()
        origin = self.data.xpos[self.target_id]
        tcp = np.array([origin[0], origin[1], max(hi - 0.012, lo + 0.014)])
        for yaw in np.arange(4) * np.pi / 4.0 - np.pi / 4.0:
            closing = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
            a, b = self.object_extent(closing)
            half = max(b - origin @ closing, origin @ closing - a)
            if half + 0.002 > GRIPPER_OPEN_WIDTH:
                continue
            if not (self.hand_collides(tcp, yaw, min(half + 0.005, GRIPPER_OPEN_WIDTH), {self.target_id})
                    or self.hand_collides(tcp, yaw, half + 0.0005, {self.target_id})):
                return True
        return False

    def overhead_clear(self, xy: np.ndarray, z: float, reach: float = 0.05, height: float = 0.20) -> bool:
        """True if no colliding fixture lies within `height` above `z` over a square of half-width `reach`
        around `xy`, i.e. the hand can approach the body top-down."""
        for dx in (-reach, 0.0, reach):
            for dy in (-reach, 0.0, reach):
                hit = self.ray_vertical(np.array([xy[0] + dx, xy[1] + dy, z]), up=True)
                if hit is not None and hit - z < height:
                    return False
        return True

    def _footprint_hits(self) -> list[float]:
        """Heights of the first surfaces below the bottom geoms of the body in its current pose, sampled at
        the center, edges and corners of each geom's footprint; rays start just above the body's top so
        overhanging fixtures (a brew head over a drip tray) are not mistaken for the support."""
        m, d = self.model, self.data
        lo, hi = self.object_extent_z()
        hits: list[float] = []
        box_grid = [(ox, oy) for ox in (-0.95, 0.0, 0.95) for oy in (-0.95, 0.0, 0.95)]
        round_grid = [(0.0, 0.0)] + [(0.9 * np.cos(a), 0.9 * np.sin(a)) for a in np.arange(8) * np.pi / 4.0]
        round_types = (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_SPHERE,
                       mujoco.mjtGeom.mjGEOM_CAPSULE)
        for g in self.object_geom_ids:
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            half = m.geom_aabb[g][3:]
            corners_z = d.geom_xpos[g][2] + (m.geom_aabb[g][:3] + half * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])) @ rot[2]
            if corners_z.min() > lo + 0.003:
                continue  # only the geoms the body stands on
            center = d.geom_xpos[g] + rot @ m.geom_aabb[g][:3]
            for ox, oy in (round_grid if m.geom_type[g] in round_types else box_grid):
                point = center + rot @ np.array([ox * half[0], oy * half[1], 0.0])
                hit = self.ray_vertical(np.array([point[0], point[1], hi + 0.01]))
                if hit is not None:
                    hits.append(hit)
        return hits

    def _support_height(self, goal: np.ndarray) -> float | None:
        """Height of the body origin when lowered onto the support under its footprint at goal xy (body
        upright), or None if nothing supports it; sets support_flatness to the spread of the support."""
        self.pin_free_body(self.target_qposadr, goal)
        mujoco.mj_forward(self.model, self.data)
        lo, _ = self.object_extent_z()
        hits = self._footprint_hits()
        if not hits:
            return None
        self.support_flatness = float(max(hits) - min(hits))
        return float(goal[2] - lo + max(hits) + 0.0005)

    def _place_scene_bodies(self) -> None:
        for jn, pos in self.scene_bodies.items():
            self.pin_free_body(self.scene_body_qposadr[jn], np.asarray(pos, dtype=np.float64))
        for adr in self.distractor_qposadr:
            self.pin_free_body(adr, self.model.qpos0[adr : adr + 3])

    def _calibrate_goal(self) -> None:
        """Drop the body at the goal to find its rest height, or mark the goal as a hold goal."""
        self._place_scene_bodies()
        self.pin_free_body(self.target_qposadr, np.array([-0.6, 0.0, -0.40]))  # out of the way
        self._settle()
        z = self._support_height(self.target_goal_pos)
        if z is not None:
            self.pin_free_body(self.target_qposadr, np.array([*self.target_goal_pos[:2], z]))
            self._settle(max_substeps=1500)
        pos = self.data.xpos[self.target_id].copy()
        drift = float(np.linalg.norm(pos[:2] - self.target_goal_pos[:2]))
        self.calibration = {"support_z": z, "rest": pos, "drift": drift, "tilt_deg": float(np.degrees(self.tilt())),
                            "graspable": self.graspable()}
        mid_air = z is None or self.target_goal_pos[2] > z + HOLD_HEIGHT
        unstable = drift > 0.005 or self.tilt() > self.tilt_tol
        if mid_air or unstable:
            self.hold_goal = True
            self.goal_xy = self.target_goal_pos[:2].copy()
            self.goal_z = float(self.target_goal_pos[2]) if z is None else max(float(self.target_goal_pos[2]), z + 0.002)
        else:
            # The goal is the settled resting pose at the target: a body dropped over a hole or into a
            # compartment comes to rest at the receptacle's center, which the success tolerance refers to.
            self.hold_goal = False
            self.goal_xy = pos[:2].copy()
            self.goal_z = float(pos[2])
        self.reset()

    def _project_press_goal(self) -> np.ndarray:
        """Reach-and-press goal the TCP can occupy: lifted out of any fixture so that the finger pads
        (PAD_BELOW_TCP below the TCP) stop 2 mm above the highest surface under the closed fingers."""
        mujoco.mj_forward(self.model, self.data)
        goal = self.target_goal_pos.copy()
        surface = -np.inf
        geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # scene geoms only, not the robot
        geom_id = np.zeros(1, dtype=np.int32)
        for dx, dy in ((0.0, 0.0), (0.0, 0.012), (0.0, -0.012), (0.012, 0.0), (-0.012, 0.0)):
            start = goal + np.array([dx, dy, 0.3])
            dist = mujoco.mj_ray(self.model, self.data, start, np.array([0.0, 0.0, -1.0]), geomgroup, 1, -1, geom_id)
            if dist >= 0:
                surface = max(surface, start[2] - dist)
        goal[2] = max(goal[2], surface + PAD_BELOW_TCP + 0.002)
        return goal

    @property
    def goal_position(self) -> np.ndarray:
        """Goal of the body's center: the calibrated resting pose at the target (or the target at the hold
        height). With a goal reference body, the goal xy is that body's axis at the goal height."""
        if self.goal_ref_id < 0:
            return np.array([self.goal_xy[0], self.goal_xy[1], self.goal_z])
        ref = self.data.xpos[self.goal_ref_id]
        axis = self.data.xmat[self.goal_ref_id].reshape(3, 3)[:, 2]
        point = ref + axis * (self.goal_z - ref[2]) / max(axis[2], 1e-6)
        return np.array([point[0], point[1], self.goal_z])

    # ----------------------------------------------------------------- Reset
    def _sample_distractor_xy(self, avoid: list[np.ndarray]) -> np.ndarray:
        for _ in range(50):
            xy = np.array([self.perturbation_engine.uniform(*WORKSPACE_X), self.perturbation_engine.uniform(*WORKSPACE_Y)])
            if all(np.linalg.norm(xy - a[:2]) > 0.09 for a in avoid):
                return xy
        return xy

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        engine = self.perturbation_engine
        engine.apply_visual_perturbation(self.model)
        if self.target_qposadr is None:
            self._place_scene_bodies()
            self._settle()
            self.dwell = 0
            self.start_fallback = False
            return

        engine.apply_object_perturbation(self.model, self.object_geom_ids)
        engine.apply_physical_perturbation(self.model, self.target_id, self.object_geom_ids, self.pad_geom_ids)
        q_arm = self.joint_positions
        # Rejection-sample the full scatter first, then shrink it so that bodies starting in a compartment
        # or on a small fixture still get a valid, varied start before the nominal fallback.
        self.start_fallback = True
        for scale, attempts in RESET_SCATTER_SCHEDULE:
            if any(self._try_start_pose(q_arm, *engine.sample_pose(self.nominal_obj_pos, tilt=self.spawn_tilt_ok, scale=scale))
                   for _ in range(attempts)):
                self.start_fallback = False
                break
        else:
            self._try_start_pose(q_arm, self.nominal_obj_pos.copy(), np.array([1.0, 0.0, 0.0, 0.0]), strict=False)
        self.start_z = float(self.data.xpos[self.target_id][2])
        self.peak_z = self.start_z
        self.dwell = 0

    def _try_start_pose(self, q_arm: np.ndarray, pos: np.ndarray, quat: np.ndarray, strict: bool = True) -> bool:
        """Place the body (and distractors) for a candidate start, settle, and report whether the start is
        valid: resting flat and upright on the table, away from the goal (unless the task returns the body
        to its nominal start), with no fixture overhead and a collision-free top-down grasp. With `strict` unset (the nominal fallback, e.g. a
        gear seated on its shaft) the body is lowered onto its support even if that support is not flat."""
        self._set_arm_state(q_arm)
        self._place_scene_bodies()
        support = self._support_height(pos)
        if strict and (support is None or self.support_flatness > 0.002):
            return False  # footprint straddles an edge or hangs off the table
        pos = np.array([pos[0], pos[1], pos[2] if support is None else support + 0.002])
        self.pin_free_body(self.target_qposadr, pos, quat)
        if self.use_distractors:
            avoid = [pos, self.target_goal_pos]
            for adr in self.distractor_qposadr:
                xy = self._sample_distractor_xy(avoid)
                yaw = self.perturbation_engine.uniform(-np.pi, np.pi)
                self.pin_free_body(adr, np.array([xy[0], xy[1], 0.06]), np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
                avoid.append(np.array([xy[0], xy[1], 0.0]))
        self._settle()
        p = self.data.xpos[self.target_id]
        # The spawn roll/pitch is resolved by settling: the body must come to rest flat on its support,
        # so starts leaning on a rim or a neighbouring body are rejected.
        upright = self.tilt() < np.radians(3.0)
        hits = self._footprint_hits()
        resting_flat = bool(hits) and max(hits) - min(hits) <= 0.002  # not bridging a rim or a gap
        on_table = p[2] > 0.0 and WORKSPACE_X[0] - 0.05 < p[0] < WORKSPACE_X[1] + 0.05
        returns_to_start = np.allclose(self.nominal_obj_pos, self.target_goal_pos)
        away_from_goal = returns_to_start or np.linalg.norm(p[:2] - self.target_goal_pos[:2]) > max(3.0 * self.tolerance_xy, 0.02)
        accessible = self.overhead_clear(p[:2], self.object_extent_z()[1] + 0.005)
        return bool(upright and resting_flat and on_table and away_from_goal and accessible and self.graspable())

    # ----------------------------------------------------------------- Step & success
    def _after_step(self) -> None:
        if self.target_qposadr is not None:
            self.peak_z = max(self.peak_z, float(self.data.xpos[self.target_id][2]))
        elif self.task_type == "press_button":
            near = np.linalg.norm(self.tcp_position - self.press_goal) <= self.tolerance_xy
            self.dwell = self.dwell + 1 if near else 0

    def object_speed(self) -> tuple[float, float]:
        dof = self.free_qpos_to_dof[self.target_qposadr]
        v = self.data.qvel[dof : dof + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def at_goal(self) -> bool:
        err = self.data.xpos[self.target_id] - self.goal_position
        return bool(np.linalg.norm(err[:2]) <= self.tolerance_xy and abs(err[2]) <= self.tolerance_z
                    and self.tilt() <= self.tilt_tol)

    def check_success(self) -> bool:
        if self.task_type == "press_button":
            return self.dwell >= PRESS_DWELL_STEPS
        if self.target_qposadr is None:
            return False
        lifted = self.peak_z - self.start_z >= self.min_lift - 1e-6
        if not (lifted and self.at_goal()):
            return False
        lin, ang = self.object_speed()
        contacts = self.body_contacts(self.target_id)
        if self.hold_goal:
            return bool(self.finger_body_ids <= contacts and lin < REST_LIN_SPEED)
        released = not (contacts & self.finger_body_ids)
        supported = bool(contacts - self.finger_body_ids - self.robot_body_ids)
        return bool(released and supported and lin < REST_LIN_SPEED and ang < REST_ANG_SPEED)
