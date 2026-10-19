"""Multi-goal Franka environment of AFFORD-Bench.

The environment builds the scene of a TaskSpec, finds at start-up the resting pose of every goal and pre-placed start
by dropping the bodies there (so goals are physically attainable and stable), samples a valid start layout at each
reset, and ends the episode at the first step at which all goals hold and every ordering constraint is met.

The single-body geometry helpers of ParametricFrankaEnv (support search, grasp feasibility, extents, speed) act on the
"focused" body; `focus` points them at any free body of the scene.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import BaseFrankaEnv, _rot_z
from afford_pro.envs.parametric_env import ASSET_DIR, PAD_BELOW_TCP, ParametricFrankaEnv
from afford_pro.perturbations.engine import (OBJECT_MODES, PHYSICAL_MODES, SPATIAL_MODES, PerturbationConfig,
                                             PerturbationEngine)
from afford_pro.bench import predicates
from afford_pro.bench.scenes import SCENES
from afford_pro.bench.spec import On, Placed, Region, Slots, TaskSpec, step_budget

PARK_ORIGIN = np.array([-0.80, -0.45, -0.45])   # floor spots behind the robot, 15 cm apart, for absent bodies
PARK_SPACING = 0.15
REGION_DROP_Z = 0.15       # pin height when dropping a body at a region center during calibration (m)
REGION_MARGIN = 0.003      # a body counts as inside a region when its footprint is this far inside the edge (m)
ANY_POSE_TILT = 90.0       # a region goal allowing this much tilt accepts the body in any pose, e.g. thrown in a bin
MAX_GOAL_DRIFT = 0.005     # a calibrated goal pose must not slide further than this while settling (m)
START_TILT = np.radians(3.0)
MAX_START_SHIFT = 0.005    # a pinned start must not move further than this while settling (m)
DISTRACTORS_PER_TASK = 2
# (scatter scale, attempts) of start-layout rejection sampling before the region-center fallback; the legacy tasks made 16 attempts
# at full scale, too few for crowded scenes, which then fell back to near-fixed starts.
RESET_SCATTER_SCHEDULE = ((1.0, 40), (0.5, 16), (0.25, 8), (0.1, 8), (0.05, 8))
SEPARATION = 0.0           # least horizontal gap between the footprint circles of two sampled start poses (m)
LIFT_RETRIES = (0.003, 0.006)  # lift a start pose that penetrates a ribbed support by this much and retry (m)
OVERHEAD_REACH = 0.03      # half-width of the column above a start that must be free of fixtures (m)
# Null-space pull of the IK toward the ready posture (per iteration). The legacy tasks' IK had none, so lifting a part close to the
# robot could fold the arm into the shoulder singularity (joint 2 near 0, joints 1 and 3 aligned), where the joint
# servos no longer realize the commanded motion and the hand drifts.
NULLSPACE_GAIN = 0.1
TABLE_X, TABLE_Y = (0.25, 0.75), (-0.42, 0.42)


class SpecError(ValueError):
    """A task specification is not physically realizable in its scene."""


class GoalFrankaEnv(ParametricFrankaEnv):
    PAD_BELOW_TCP = PAD_BELOW_TCP

    def __init__(self, spec: TaskSpec, perturbation_mode: str = "nominal", render_mode: str = "rgb_array",
                 max_episode_steps: int | None = None, **kwargs: Any):
        self.task_spec = spec
        self.scene = SCENES[spec.scene]
        self.scene_xml_path = ASSET_DIR / self.scene.xml
        # Attributes the ParametricFrankaEnv helpers read.
        self.task_type = "goal"
        self.tolerance_xy, tilt_deg = self.scene.precision
        self.tolerance_z = self.tolerance_xy
        self.tilt_tol = np.radians(tilt_deg)
        self.min_lift = 0.02
        self.init_tcp_pos = None
        self.scene_bodies: dict[str, list[float]] = {}
        self.use_distractors = False
        self.goal_ref_body = None
        self.hold_goal = False
        self.perturbation_engine = PerturbationEngine(PerturbationConfig(mode=perturbation_mode))
        self._calibrated = False
        BaseFrankaEnv.__init__(self, xml_path=self.scene_xml_path, render_mode=render_mode,
                               max_episode_steps=max_episode_steps or step_budget(spec), **kwargs)
        self.language_instruction = spec.instruction
        self._calibrate()
        self._calibrated = True
        self.reset()

    # ----------------------------------------------------------------- Setup
    def _setup_task(self) -> None:
        m = self.model
        body_id = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
        joint_adr = lambda name: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])  # noqa: E731
        self.body_ids = {name: body_id(name) for name in self.scene.bodies}
        missing = [n for n, b in self.body_ids.items() if b < 0]
        if missing:
            raise SpecError(f"{self.scene.xml} has no body {missing}")
        self.body_qposadr = {n: self.free_body_qposadr[b] for n, b in self.body_ids.items()}
        self.body_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b] for n, b in self.body_ids.items()}
        self.pad_geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.finger_body_ids and m.geom_contype[g]]
        self.button_adr = {n: joint_adr(b.joint) for n, b in self.scene.buttons.items()}
        self.button_cap = {n: body_id(b.cap) for n, b in self.scene.buttons.items()}
        self.slider_adr = {n: joint_adr(s.joint) for n, s in self.scene.sliders.items()}
        self.slider_face = {n: body_id(s.face) for n, s in self.scene.sliders.items()}
        self.other_free_adr = [a for b, a in sorted(self.free_body_qposadr.items()) if b not in self.body_ids.values()]
        self.goal_body_offset: dict[tuple[str, str], np.ndarray] = {}
        self.goal_site_rest: dict[tuple[str, str], np.ndarray] = {}
        self.goal_site_yaw: dict[tuple[str, str], float] = {}
        self.goal_site_floor: dict[tuple[str, str], float] = {}   # height of the body's lowest point at its goal rest
        self.start_rest: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        # Targets that move with a mechanism (inside a drawer): the mechanism body's position at calibration.
        self.attach_id = {t: body_id(b) for t, b in self.scene.attached.items()}
        self.attach_ref: dict[str, np.ndarray] = {}
        self.body_radius = {n: self._footprint_radius(b) for n, b in self.body_ids.items()}
        self.present = self._present_bodies()
        self.focus(next(iter(self.scene.bodies)))
        self.park_pos = {n: self._resting_park_pos(n, k) for k, n in enumerate(self.body_ids)}
        self._begin_bookkeeping()

    def _present_bodies(self) -> tuple[str, ...]:
        """Bodies in this task's scene: those its goals and starts name (and the bodies they rest on), plus clutter."""
        names: list[str] = []

        def add(name: str | None) -> None:
            if name in self.body_ids and name not in names:
                names.append(name)
                start = self.start_of(name)
                add(start.target if isinstance(start, Placed) else None)

        for goal in self.task_spec.goals:
            add(getattr(goal, "body", None))
            add(getattr(goal, "target", None))
        for body in self.task_spec.starts:
            add(body)
        extra = self.task_spec.extra_bodies
        if extra is None:
            extra = tuple(b for b in self.scene.distractors if b not in names)[:DISTRACTORS_PER_TASK]
        for body in extra:
            add(body)
        return tuple(b for b in self.scene.bodies if b in names)

    def _resting_park_pos(self, body: str, slot: int) -> np.ndarray:
        """Where an absent body lies at rest on the floor behind the robot (so it never moves during a reset)."""
        spot = PARK_ORIGIN + np.array([0.0, PARK_SPACING * slot, 0.10])
        self.pin_free_body(self.body_qposadr[body], spot)
        mujoco.mj_forward(self.model, self.data)
        lowest = self.body_bottom(body)
        return spot + np.array([0.0, 0.0, PARK_ORIGIN[2] - lowest + 0.0005])

    def _footprint_radius(self, body_id: int) -> float:
        """Largest horizontal distance from the body origin to its collision geometry (body upright)."""
        m = self.model
        radius = 0.0
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body_id or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            rot = rot.reshape(3, 3)
            round_types = (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID)
            if m.geom_type[g] in round_types and rot[2, 2] > 0.999:
                ellipsoid = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
                reach = max(m.geom_size[g][0], m.geom_size[g][1]) if ellipsoid else m.geom_size[g][0]
                radius = max(radius, float(np.linalg.norm(m.geom_pos[g][:2]) + reach))
                continue
            corners = m.geom_pos[g] + rot @ m.geom_aabb[g][:3] + (signs * m.geom_aabb[g][3:]) @ rot.T
            radius = max(radius, float(np.linalg.norm(corners[:, :2], axis=1).max()))
        return radius

    def focus(self, body: str) -> None:
        """Point the single-body helpers (extents, support search, grasp checks, speed) at `body`."""
        self.object_name = body
        self.target_id = self.body_ids[body]
        self.target_qposadr = self.body_qposadr[body]
        self.object_geom_ids = self.body_geoms[body]
        self.start_z = self.start_z_of.get(body, float(self.data.xpos[self.target_id][2])) if hasattr(self, "start_z_of") else 0.0
        self.focused_goal: int | None = None

    def focus_goal(self, index: int) -> None:
        goal = self.task_spec.goals[index]
        self.focus(goal.body)
        self.focused_goal = index
        self.min_lift = goal.min_lift if isinstance(goal, On) else 0.0

    # ----------------------------------------------------------------- Kinematics
    def solve_ik(self, target_pos: np.ndarray, target_yaw: float, q_init: np.ndarray, max_iter: int = 60,
                 tol: float = 1e-4) -> np.ndarray:
        """The base env's damped-least-squares IK (TCP position, hand pointing down with the given yaw) plus a
        null-space pull toward the ready posture, which keeps the redundant arm away from singular folds."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        for i, adr in enumerate(self.arm_qpos_adr):
            d.qpos[adr] = q_init[i]
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        r_target = _rot_z(target_yaw) @ self._r_down
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(max_iter):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            tcp = self._tcp_from(d)
            r_cur = d.xmat[self.hand_id].reshape(3, 3)
            err_p = target - tcp
            err_r = 0.5 * sum(np.cross(r_cur[:, k], r_target[:, k]) for k in range(3))
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 1e-3:
                break
            mujoco.mj_jac(m, d, jacp, jacr, tcp, self.hand_id)
            jac = np.vstack([jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]])
            jjt = jac @ jac.T + 1e-4 * np.eye(6)
            q = np.array([d.qpos[adr] for adr in self.arm_qpos_adr])
            dq = jac.T @ np.linalg.solve(jjt, np.concatenate([err_p, 0.5 * err_r]))
            dq += (np.eye(7) - jac.T @ np.linalg.solve(jjt, jac)) @ (NULLSPACE_GAIN * (self.ready_q - q))
            for i, adr in enumerate(self.arm_qpos_adr):
                d.qpos[adr] = np.clip(d.qpos[adr] + dq[i], *self.arm_joint_range[i])
        return np.array([d.qpos[adr] for adr in self.arm_qpos_adr])

    # ----------------------------------------------------------------- Geometry
    def _attachment_shift(self, target: str) -> np.ndarray:
        """How far the mechanism carrying `target` has moved since calibration (zero for fixed targets)."""
        if target not in self.attach_id or target not in self.attach_ref:
            return np.zeros(3)
        return self.data.xpos[self.attach_id[target]] - self.attach_ref[target]

    def target_point(self, name: str) -> np.ndarray:
        if name in self.scene.targets:
            return np.asarray(self.scene.targets[name], dtype=np.float64) + self._attachment_shift(name)
        return self.data.xpos[self.body_ids[name]].copy()

    def on_tolerances(self, goal: On) -> tuple[float, float, float]:
        """(xy, z, tilt) tolerances: the goal's own, else for a region target the room left for the body's footprint
        inside the region, else the scene's precision."""
        tol, tilt = self.scene.precision
        tol_xy = goal.tol_xy
        if tol_xy is None and goal.target in self.scene.regions:
            half = self.scene.regions[goal.target][1]
            footprint = 0.0 if self.any_pose(goal) else self.body_radius[goal.body]  # in any pose: its origin inside
            tol_xy = max(tol, half - footprint - REGION_MARGIN)
        return (tol if tol_xy is None else tol_xy, tol if goal.tol_z is None else goal.tol_z,
                np.radians(tilt if goal.tilt_deg is None else goal.tilt_deg))

    def region_center(self, goal: On) -> np.ndarray | None:
        if goal.target not in self.scene.regions:
            return None
        return self.target_point(self.scene.regions[goal.target][0])

    def goal_point(self, goal: On) -> np.ndarray:
        """Where the body's origin rests when `goal` holds: the calibrated rest pose at a target point, or the
        calibrated offset from the current pose of the body it is placed on."""
        if goal.target in self.scene.targets:
            return self.goal_site_rest[(goal.body, goal.target)] + self._attachment_shift(goal.target)
        return self.data.xpos[self.body_ids[goal.target]] + self.goal_body_offset[(goal.body, goal.target)]

    def any_pose(self, goal: On) -> bool:
        return goal.target in self.scene.regions and goal.tilt_deg is not None and goal.tilt_deg >= ANY_POSE_TILT

    def on_error(self, goal: On) -> np.ndarray:
        """Body position minus its goal rest: in xy from the region center for a region target; in z, for a goal that
        accepts any pose, the body's lowest point above the region's floor (a sponge on its side is in the bin)."""
        pos = self.data.xpos[self.body_ids[goal.body]]
        err = pos - self.goal_point(goal)
        center = self.region_center(goal)
        if center is not None:
            err[:2] = pos[:2] - center[:2]
        if self.any_pose(goal):
            floor = self.goal_site_floor[(goal.body, goal.target)] + self._attachment_shift(goal.target)[2]
            err[2] = self.body_bottom(goal.body) - floor
        return err

    def on_near(self, goal: On) -> bool:
        tol_xy, tol_z, tilt = self.on_tolerances(goal)
        err = self.on_error(goal)
        return bool(np.linalg.norm(err[:2]) <= tol_xy and abs(err[2]) <= tol_z and self.body_tilt(goal.body) <= tilt)

    def body_bottom(self, body: str) -> float:
        focused = self.object_geom_ids
        self.object_geom_ids = self.body_geoms[body]
        try:
            return self.object_extent_z()[0]
        finally:
            self.object_geom_ids = focused

    def surface_top(self, target: str) -> float:
        """Height of the top of a target: a free body's highest point, or the first surface below a target point."""
        if target in self.body_ids:
            return self.body_top(target)
        point = self.target_point(target)
        hit = self.ray_vertical(np.array([point[0], point[1], 0.6]))
        return float(point[2]) if hit is None else hit

    @property
    def goal_position(self) -> np.ndarray:
        goal = None if self.focused_goal is None else self.task_spec.goals[self.focused_goal]
        return self.goal_point(goal) if isinstance(goal, On) else self.data.xpos[self.target_id].copy()

    def body_tilt(self, body: str) -> float:
        """Angle (rad) of the body away from its resting attitude: of its up axis from vertical (any face for cubes),
        or of its lying axis from horizontal (a cylinder on its side may roll freely about that axis)."""
        rot = self.data.xmat[self.body_ids[body]].reshape(3, 3)
        info = self.scene.bodies[body]
        if info.lies_along is not None:
            axis = "xyz".index(info.lies_along)
            return float(np.arcsin(np.clip(abs(rot[2, axis]), 0.0, 1.0)))
        cos = np.abs(rot[2]).max() if info.any_face_up else rot[2, 2]
        return float(np.arccos(np.clip(cos, -1.0, 1.0)))

    def tilt(self) -> float:
        return self.body_tilt(self.object_name)

    def body_yaw(self, body: str) -> float:
        rot = self.data.xmat[self.body_ids[body]].reshape(3, 3)
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def goal_yaw(self, goal: On) -> float | None:
        """Yaw of the body when resting at a target point (None for goals on another body)."""
        return self.goal_site_yaw.get((goal.body, goal.target))

    def body_speed(self, body: str) -> tuple[float, float]:
        dof = self.free_qpos_to_dof[self.body_qposadr[body]]
        v = self.data.qvel[dof: dof + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def _top_of_geoms(self, geoms: list[int]) -> float:
        focused = self.object_geom_ids
        self.object_geom_ids = geoms
        try:
            return self.object_extent_z()[1]
        finally:
            self.object_geom_ids = focused

    def body_top(self, body: str) -> float:
        """World z of the body's highest point (the focus is left unchanged)."""
        return self._top_of_geoms(self.body_geoms[body])

    def body_top_of_id(self, body_id: int) -> float:
        return self._top_of_geoms([g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id])

    def slider_face_point(self, name: str) -> np.ndarray:
        bid = self.slider_face[name]
        local = np.asarray(self.scene.sliders[name].face_point, dtype=np.float64)
        return self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ local

    def button_depth(self, name: str) -> float:
        return float(self.data.qpos[self.button_adr[name]])

    def slider_position(self, name: str) -> float:
        return float(self.data.qpos[self.slider_adr[name]])

    # ----------------------------------------------------------------- Calibration
    def _park_all(self) -> None:
        for body, pos in self.park_pos.items():
            self.pin_free_body(self.body_qposadr[body], pos)
        for adr in self.other_free_adr:
            self.pin_free_body(adr, self.model.qpos0[adr: adr + 3])

    def _drop_at_point(self, body: str, point: np.ndarray) -> np.ndarray:
        self.focus(body)
        z = self._support_height(np.asarray(point, dtype=np.float64))
        if z is None:
            raise SpecError(f"{self.task_spec.task_id}: nothing supports {body} at {np.round(point, 3)}")
        self.pin_free_body(self.target_qposadr, np.array([point[0], point[1], z]))
        self._settle(max_substeps=1500)
        return self._checked_rest(body, np.asarray(point[:2]))

    def _drop_on_body(self, body: str, base: str) -> np.ndarray:
        """Slide `body` down the axis of `base` (kinematically, 1 mm steps) from above its top to the first contact,
        so parts with a close-fitting bore go over a shaft instead of catching on its rim, then let it settle."""
        base_id = self.body_ids[base]
        origin = self.data.xpos[base_id].copy()
        axis = self.data.xmat[base_id].reshape(3, 3)[:, 2].copy()
        quat = self.data.xquat[base_id].copy()
        self.focus(base)
        base_top = self.object_extent_z()[1]
        self.focus(body)
        self.pin_free_body(self.target_qposadr, origin, quat)
        mujoco.mj_forward(self.model, self.data)
        clearance = base_top - self.object_extent_z()[0] + 0.003  # lift until the body's bottom clears the base
        height = clearance
        while True:
            if height < clearance - 0.30:
                raise SpecError(f"{self.task_spec.task_id}: {body} slid 30 cm down {base} without touching anything")
            self.pin_free_body(self.target_qposadr, origin + axis * (height - 0.001), quat)
            mujoco.mj_forward(self.model, self.data)
            if self._penetrates(body, depth=0.0002):
                break
            height -= 0.001
        self.pin_free_body(self.target_qposadr, origin + axis * height, quat)
        self._settle(max_substeps=1500)
        return self._checked_rest(body, self.data.xpos[base_id][:2].copy())

    def _checked_rest(self, body: str, aim_xy: np.ndarray) -> np.ndarray:
        pos = self.data.xpos[self.body_ids[body]].copy()
        drift = float(np.linalg.norm(pos[:2] - aim_xy))
        if drift > MAX_GOAL_DRIFT or self.body_tilt(body) > np.radians(self.scene.precision[1]):
            raise SpecError(f"{self.task_spec.task_id}: {body} is not stable there (drift {drift * 1e3:.1f} mm, "
                            f"tilt {np.degrees(self.body_tilt(body)):.1f} deg)")
        return pos

    def _anchor_of(self, body: str, config: dict[str, str]) -> str | None:
        return config.get(body)

    def _build(self, config: dict[str, str]) -> dict[str, np.ndarray]:
        """Rest positions of the bodies in `config` (body -> target point or base body), bases first; bases that
        are not in `config` rest at their Placed start or at their home region's center."""
        self._park_all()
        placed: dict[str, np.ndarray] = {}

        def place(body: str, chain: tuple[str, ...] = ()) -> None:
            if body in placed:
                return
            if body in chain:
                raise SpecError(f"{self.task_spec.task_id}: circular placement {chain + (body,)}")
            anchor = config.get(body) or self._default_anchor(body)
            if anchor in self.body_ids:
                place(anchor, chain + (body,))
                placed[body] = self._drop_on_body(body, anchor)
            elif anchor in self.scene.targets:
                placed[body] = self._drop_at_point(body, self.target_point(anchor))
            else:
                region = self.start_of(body)
                placed[body] = self._drop_at_point(body, np.array([*region.center, REGION_DROP_Z]))

        for body in config:
            place(body)
        return placed

    def _record_goal_rest(self, rest: dict[str, np.ndarray], config: dict[str, str]) -> None:
        for body, target in config.items():
            if target in self.scene.targets:
                self.goal_site_rest[(body, target)] = rest[body]
                self.goal_site_yaw[(body, target)] = self.body_yaw(body)
                self.goal_site_floor[(body, target)] = self.body_bottom(body)
            else:
                self.goal_body_offset[(body, target)] = rest[body] - rest[target]

    def _default_anchor(self, body: str) -> str | None:
        start = self.start_of(body)
        return start.target if isinstance(start, Placed) else None

    def start_of(self, body: str):
        return self.task_spec.starts.get(body, self.scene.bodies[body].home)

    def _calibrate_goal(self) -> None:
        """Re-calibrate after the model was reloaded (robomimic's reset_from_xml_string calls this hook)."""
        self._calibrated = False
        self._calibrate()
        self._calibrated = True

    def _calibrate(self) -> None:
        q_arm = self.joint_positions
        self._set_arm_state(q_arm)
        for name, adr in self.slider_adr.items():
            self.data.qpos[adr] = self.scene.sliders[name].open  # attached targets are given with sliders open
        mujoco.mj_forward(self.model, self.data)
        self.attach_ref = {t: self.data.xpos[b].copy() for t, b in self.attach_id.items()}
        placements = [g for g in self.task_spec.goals if isinstance(g, On)]
        final = {g.body: g.target for g in placements}  # a body's last placement goal; earlier ones are waypoints
        self._record_goal_rest(self._build(final), final)
        for g in placements:
            if final[g.body] != g.target:
                self._record_goal_rest(self._build({g.body: g.target}), {g.body: g.target})
        placed_starts = {b: s.target for b in self.present if isinstance(s := self.start_of(b), Placed)}
        if placed_starts:
            self._build(placed_starts)
            for body, target in placed_starts.items():
                bid = self.body_ids[body]
                self.start_rest[(body, target)] = (self.data.xpos[bid].copy(), self.data.xquat[bid].copy())
        for body in self.present:
            start = self.start_of(body)
            if isinstance(start, Slots):
                for slot in start.targets:
                    self._build({body: slot})
                    bid = self.body_ids[body]
                    self.start_rest[(body, slot)] = (self.data.xpos[bid].copy(), self.data.xquat[bid].copy())
        self.focus(next(iter(self.scene.bodies)))

    # ----------------------------------------------------------------- Reset
    def _begin_bookkeeping(self) -> None:
        self.start_pos_of = {b: self.data.xpos[i].copy() for b, i in self.body_ids.items()}
        # Where each goal's body was when the goal became due (at the start, or when the goals ordered before it were
        # reached): a touched body must stay within 1 cm of it.
        self.active_pos = {i: self.start_pos_of[g.body].copy() for i, g in enumerate(self.task_spec.goals)
                           if hasattr(g, "body") and not self._predecessors(i)}
        self.start_z_of = {b: float(p[2]) for b, p in self.start_pos_of.items()}
        self.peak_z = dict(self.start_z_of)
        self.dwell = {i: 0 for i in range(len(self.task_spec.goals))}
        self.first_true: dict[int, int] = {}
        # Goals seen not holding while due: a goal is reached only on such a rising edge, so a latching button pressed
        # before the goals ordered ahead of it, or a part already in place, does not count when they complete. Goals
        # due from the start are armed at once (start layouts are sampled with none of them holding).
        self.armed: set[int] = {i for i in range(len(self.task_spec.goals)) if not self._predecessors(i)}
        self.start_z = self.start_z_of.get(self.object_name, 0.0)

    def _perturb_bodies(self) -> None:
        """Color (p1, L5) and friction/mass (L4, L5) of every body the task moves; one friction draw per episode."""
        engine, cfg = self.perturbation_engine, self.perturbation_engine.config
        moved = self.task_spec.bodies_moved()
        if engine.mode in OBJECT_MODES:
            for body in moved:
                engine.apply_object_perturbation(self.model, self.body_geoms[body])
        if engine.mode in PHYSICAL_MODES:
            mu = engine.model_rng.uniform(*cfg.friction_range)
            for g in self.pad_geom_ids + [g for b in moved for g in self.body_geoms[b]]:
                self.model.geom_friction[g][0] = mu
            for body in moved:
                scale = engine.model_rng.uniform(*cfg.mass_scale_range)
                self.model.body_mass[self.body_ids[body]] *= scale
                self.model.body_inertia[self.body_ids[body]] *= scale

    def _reset_mechanisms(self) -> None:
        for name, adr in self.button_adr.items():
            self.data.qpos[adr] = 0.0
        for name, adr in self.slider_adr.items():
            self.data.qpos[adr] = self.task_spec.sliders.get(name, self.scene.sliders[name].open)

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        if not self._calibrated:
            self._park_all()
            return
        self.perturbation_engine.apply_visual_perturbation(self.model)
        self._perturb_bodies()
        q_arm = self.joint_positions
        self.start_fallback = True
        self.layout_rejections: Counter[str] = Counter()  # why candidate start layouts were rejected (diagnostics)
        for scale, attempts in RESET_SCATTER_SCHEDULE:
            if any(self._try_layout(q_arm, scale) for _ in range(attempts)):
                self.start_fallback = False
                break
        else:
            problem = self._layout_problem(q_arm, 0.0, strict=False)
            if problem is not None:
                raise RuntimeError(f"{self.task_spec.task_id}: no start layout, even at the region centers ({problem})")
        self._begin_bookkeeping()

    def _layout_order(self) -> list[str]:
        """Pre-placed bodies first (bases before the parts on them), then the bodies the task moves, then the rest."""
        placed = [b for b in self.present if isinstance(self.start_of(b), Placed)]
        placed.sort(key=lambda b: self.start_of(b).target in self.body_ids)
        moved = [b for b in self.task_spec.bodies_moved() if b not in placed]
        return placed + moved + [b for b in self.present if b not in placed and b not in moved]

    def _try_layout(self, q_arm: np.ndarray, scale: float, strict: bool = True) -> bool:
        problem = self._layout_problem(q_arm, scale, strict)
        if problem is not None:
            self.layout_rejections[problem] += 1
        return problem is None

    def _layout_problem(self, q_arm: np.ndarray, scale: float, strict: bool) -> str | None:
        """Place every present body for one candidate start; None if the layout is valid, else why it is not."""
        self._set_arm_state(q_arm)
        self._park_all()
        self._reset_mechanisms()
        pinned: dict[str, np.ndarray] = {}
        for body in self._layout_order():
            pose = self._sample_start(body, scale, strict)
            if pose is None:
                return f"no flat support: {body}"
            if strict and not self._separated(body, pose[0], pinned):
                return f"too close to another body: {body}"
            if not self._pin_clear(body, pose) and strict:
                return f"intersects the scene: {body}"
            pinned[body] = pose[0]
        self._settle()
        return None if not strict else self._layout_invalid(pinned)

    def _separated(self, body: str, pos: np.ndarray, pinned: dict[str, np.ndarray]) -> bool:
        """A region start keeps SEPARATION between its footprint and every body already placed (so that no body is
        sampled on top of another); receptacle starts are exempt."""
        if not isinstance(self.start_of(body), Region):
            return True
        return all(np.linalg.norm(pos[:2] - other[:2]) >= self.body_radius[body] + self.body_radius[b] + SEPARATION
                   for b, other in pinned.items())

    def _pin_clear(self, body: str, pose: tuple[np.ndarray, np.ndarray]) -> bool:
        """Pin the body; if it penetrates something (a rib the support rays missed), lift it a little and retry."""
        pos, quat = pose
        for lift in (0.0, *LIFT_RETRIES):
            self.pin_free_body(self.body_qposadr[body], pos + np.array([0.0, 0.0, lift]), quat)
            mujoco.mj_forward(self.model, self.data)
            if not self._penetrates(body):
                return True
        return False

    def _fixtures_clear_above(self, body: str) -> bool:
        """No fixture (anything but the robot and the free bodies) lies within 20 cm above the body's top over a
        square of half-width OVERHEAD_REACH: the hand can approach from above."""
        m, d = self.model, self.data
        free_roots = set(self.body_ids.values())
        top = self.body_top(body)
        center = d.xpos[self.body_ids[body]]
        geom_id = np.zeros(1, dtype=np.int32)
        for dx in (-OVERHEAD_REACH, 0.0, OVERHEAD_REACH):
            for dy in (-OVERHEAD_REACH, 0.0, OVERHEAD_REACH):
                point = np.array([center[0] + dx, center[1] + dy, top + 0.005])
                for _ in range(8):
                    dist = mujoco.mj_ray(m, d, point, np.array([0.0, 0.0, 1.0]), None, 1, self.body_ids[body], geom_id)
                    if dist < 0 or dist > 0.20:
                        break
                    g = int(geom_id[0])
                    owner = int(m.body_rootid[m.geom_bodyid[g]])
                    solid = m.geom_contype[g] or m.geom_conaffinity[g]
                    if solid and owner != self.robot_root_id and int(m.geom_bodyid[g]) not in free_roots:
                        return False
                    point = point + np.array([0.0, 0.0, dist + 1e-4])
        return True

    def _sample_start(self, body: str, scale: float, strict: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        start, engine = self.start_of(body), self.perturbation_engine
        if isinstance(start, Placed):
            return self.start_rest[(body, start.target)]
        if isinstance(start, Slots):
            slot = start.targets[int(engine.rng.integers(len(start.targets)))]
            pos, quat = self.start_rest[(body, slot)]
            return pos, _yawed(quat, np.radians(engine.uniform(-start.yaw_deg, start.yaw_deg)))
        center = np.array([*start.center, REGION_DROP_Z])
        if engine.mode in SPATIAL_MODES and body in self.task_spec.bodies_moved():
            pos, quat = engine.sample_pose(center, tilt=False, scale=scale)
        else:
            pos = center + np.array([*(engine.uniform(-1.0, 1.0, size=2) * np.array(start.half) * scale), 0.0])
            quat = _yawed(np.array([1.0, 0.0, 0.0, 0.0]), np.radians(engine.uniform(-start.yaw_deg, start.yaw_deg)))
        self.focus(body)
        support = self._support_height(pos)
        if support is None or (strict and self.support_flatness > self.scene.bodies[body].support_spread):
            return None
        return np.array([pos[0], pos[1], support + 0.002]), quat

    def _penetrates(self, body: str, depth: float = 0.001) -> bool:
        bid = self.body_ids[body]
        for c in self.data.contact[: self.data.ncon]:
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if bid in (b1, b2) and c.dist < -depth:
                return True
        return False

    def _layout_invalid(self, pinned: dict[str, np.ndarray]) -> str | None:
        """Why a settled start layout is not valid (None if it is)."""
        moved = set(self.task_spec.bodies_moved())
        for body, pos in pinned.items():
            now = self.data.xpos[self.body_ids[body]]
            if np.linalg.norm(now[:2] - pos[:2]) > MAX_START_SHIFT:
                return f"moved while settling: {body}"
            if not (TABLE_X[0] < now[0] < TABLE_X[1] and TABLE_Y[0] < now[1] < TABLE_Y[1]):
                return f"off the table: {body}"
            if not isinstance(self.start_of(body), Region):
                continue  # pre-placed and slot starts sit in receptacles; the oracle runs show they can be grasped
            if self.body_tilt(body) > START_TILT:
                return f"not upright: {body}"
            if body in moved and self.scene.bodies[body].check_grasp:
                self.focus(body)
                if not self._fixtures_clear_above(body):
                    return f"fixture overhead: {body}"
                if not self._grasp_exists():
                    return f"no collision-free grasp: {body}"
        # Goals that follow another goal ("... then put it back") may hold at the start; they only count later.
        for i, goal in enumerate(self.task_spec.goals):
            if not self._predecessors(i) and self._goal_satisfied_at_start(i, goal):
                return f"goal {i} already holds"
        return None

    def _predecessors(self, index: int) -> list[int]:
        return [i for i, j in self.task_spec.order if j == index]

    def _grasp_exists(self) -> bool:
        from afford_pro.bench.oracle import GoalOracle  # the oracle module imports this one
        return GoalOracle(self).grasp_exists()

    def _goal_satisfied_at_start(self, index: int, goal) -> bool:
        """Whether a start already (nearly) achieves a goal: inside a region target, within 3 tolerances (at least
        2 cm) of a precise target, or the goal's state already holding."""
        if isinstance(goal, On) and self.region_center(goal) is not None:
            inside = self.on_tolerances(goal)[0] + 0.01
            return bool(np.linalg.norm(self.data.xpos[self.body_ids[goal.body]][:2] - self.region_center(goal)[:2])
                        <= inside)
        if isinstance(goal, On):
            tol_xy = self.on_tolerances(goal)[0]
            far = max(3.0 * tol_xy, 0.02)
            return bool(np.linalg.norm(self.data.xpos[self.body_ids[goal.body]][:2] - self.goal_point(goal)[:2]) <= far)
        if isinstance(goal, predicates.DWELL_GOALS):
            return predicates.dwell_condition(self, goal)
        return predicates.state_holds(self, goal)

    def set_sim_state(self, state: np.ndarray) -> None:
        BaseFrankaEnv.set_sim_state(self, state)
        if self._calibrated:
            self._begin_bookkeeping()

    # ----------------------------------------------------------------- Step & success
    def _after_step(self) -> None:
        for body, bid in self.body_ids.items():
            self.peak_z[body] = max(self.peak_z[body], float(self.data.xpos[bid][2]))
        for i, goal in enumerate(self.task_spec.goals):
            # A goal is due once every goal ordered before it has been reached.
            ready = all(p in self.first_true for p in self._predecessors(i))
            if ready and i not in self.active_pos and hasattr(goal, "body"):
                self.active_pos[i] = self.data.xpos[self.body_ids[goal.body]].copy()
            if isinstance(goal, predicates.DWELL_GOALS):
                due = ready and predicates.dwell_condition(self, goal)
                self.dwell[i] = self.dwell[i] + 1 if due else 0
            if i in self.first_true or not ready:
                continue
            # Reached on the first step it holds after having been seen not holding while due (a rising edge).
            if not self.goal_holds(i):
                self.armed.add(i)
            elif i in self.armed:
                self.first_true[i] = self.current_step

    def goal_holds(self, index: int) -> bool:
        goal = self.task_spec.goals[index]
        if getattr(goal, "once", False) and index in self.first_true:
            return True
        if isinstance(goal, predicates.DWELL_GOALS):
            return self.dwell[index] >= goal.dwell
        return predicates.state_holds(self, goal)

    def goal_report(self, index: int) -> dict[str, Any]:
        """Why a placement goal does or does not hold (errors in mm and degrees), for failure analysis."""
        goal = self.task_spec.goals[index]
        out: dict[str, Any] = {"goal": type(goal).__name__, "holds": self.goal_holds(index)}
        if isinstance(goal, On):
            bid = self.body_ids[goal.body]
            err = self.on_error(goal)
            contacts = self.body_contacts(bid)
            out.update(body=goal.body, target=goal.target, tol_mm=round(self.on_tolerances(goal)[0] * 1e3, 1),
                       xy_err_mm=round(float(np.linalg.norm(err[:2])) * 1e3, 1), z_err_mm=round(float(err[2]) * 1e3, 1),
                       tilt_deg=round(float(np.degrees(self.body_tilt(goal.body))), 1),
                       held=bool(contacts & self.finger_body_ids),
                       lifted_mm=round((self.peak_z[goal.body] - self.start_z_of[goal.body]) * 1e3, 1))
        return out

    def goal_status(self) -> list[bool]:
        return [self.goal_holds(i) for i in range(len(self.task_spec.goals))]

    def check_success(self) -> bool:
        """Every goal holds now and has been reached, each one only after the goals ordered before it."""
        if not self._calibrated:
            return False
        return all(i in self.first_true and self.goal_holds(i) for i in range(len(self.task_spec.goals)))


def _yawed(quat: np.ndarray, yaw: float) -> np.ndarray:
    """`quat` rotated by `yaw` about the world z axis."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]), np.asarray(quat, dtype=np.float64))
    return out
