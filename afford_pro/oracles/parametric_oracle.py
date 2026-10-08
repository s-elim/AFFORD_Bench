"""Closed-loop oracle for ParametricFrankaEnv tasks."""
from __future__ import annotations

import mujoco
import numpy as np

from afford_pro.envs.base_franka_env import GRIPPER_OPEN_WIDTH
from afford_pro.oracles.base_oracle import BaseOracle

OPEN, CLOSE = 1.0, -1.0
# Rotational symmetry (rad) of the grasp about the vertical axis; bodies not listed are round.
GRASP_SYMMETRY = {"red_block": np.pi / 2, "blue_block": np.pi / 2, "cheese_slice": np.pi / 2,
                  "micropipette": np.pi / 2, "coffee_mug": np.pi}
# Grasp height relative to the body origin for bodies whose top is not a good grasp (m).
GRASP_HEIGHT_OVERRIDE = {"micropipette": 0.045, "dowel_pin": 0.020}  # pin: by the head, so the fingers clear deep holes
PAD_BELOW_TCP = 0.0115   # finger pads reach this far below the TCP (m)
PAD_ABOVE_TCP = 0.006    # and this far above it (m)
FINGER_INSET = 0.0025    # the fingers' upper links reach this much further inward than the pads (m)
PALM_ABOVE_TOP = 0.022   # deepest grasp: TCP this far below the body top keeps the palm 5 mm clear (m)
CARRY_CLEARANCE = 0.12   # carry height above the higher of the pick and place grasp heights (m)
# Per-finger opening beyond the body's half-width during the approach, widest first (m); narrower
# margins let the fingers enter tight compartments.
PRESHAPE_MARGINS = (0.008, 0.005, 0.003)
YAW_LIMIT = 2.0          # largest hand yaw (rad) commanded during a carry, inside the joint-7 range
JAM_RETRIES = 3          # lift-and-realign attempts when an insertion stalls above the goal


class GraspLost(Exception):
    """The body left the fingers while being carried."""


def _grip_command(finger: float) -> float:
    """Gripper action that commands the given per-finger opening (m)."""
    return float(np.clip(2.0 * finger / GRIPPER_OPEN_WIDTH - 1.0, -1.0, 1.0))


def _wrap_half_turn(yaw: float) -> float:
    """The closing axis is symmetric under pi: map a yaw into (-pi/2, pi/2] for the joint-7 range."""
    return (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0


class ParametricOracle(BaseOracle):
    """Privileged closed-loop expert: re-reads the body pose every step, chooses a grasp yaw and finger
    pre-shape whose hand pose is collision-free (checked by IK and collision detection on a scratch
    state), verifies a two-finger grasp, corrects for the in-hand offset when placing, recovers from
    stalled insertions, and lowers the body until it is supported before releasing."""

    def run(self) -> None:
        if self.env.task_type == "press_button":
            self._press()
        else:
            for _ in range(3):
                try:
                    if self._pick():
                        self._place()
                        break
                except GraspLost:
                    pass
                self.servo(self.env.tcp_position + np.array([0.0, 0.0, 0.06]), None, OPEN, max_steps=15)
        while True:  # let the body come to rest; the episode ends on success or at the step budget
            self.hold(OPEN, 1)

    # ----------------------------------------------------------------- Helpers
    def _obj(self) -> np.ndarray:
        return self.env.data.xpos[self.env.target_id].copy()

    def _preshape(self, yaw: float, margin: float) -> float:
        """Per-finger opening for the approach: the body's half-width along the closing axis plus `margin`."""
        closing = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        lo, hi = self.env.object_extent(closing)
        c = self._obj() @ closing
        return float(min(GRIPPER_OPEN_WIDTH, max(hi - c, c - lo) + margin))

    def _hand_free(self, tcp: np.ndarray, yaw: float, finger: float) -> bool:
        return not self.env.hand_collides(tcp, yaw, finger, {self.env.target_id})

    def _grasp_candidates(self) -> list[float]:
        """Grasp yaws matching the body's symmetry, mapped into the joint-7 range."""
        env = self.env
        sym = GRASP_SYMMETRY.get(env.object_name)
        if sym is None:
            candidates = [k * np.pi / 4.0 for k in (-1, 0, 1, 2)]
        else:
            rot = env.data.xmat[env.target_id].reshape(3, 3)
            base = float(np.arctan2(rot[1, 0], rot[0, 0]))
            candidates = [base + k * sym for k in range(max(1, int(round(np.pi / sym))))]
        return sorted({round(_wrap_half_turn(c), 9) for c in candidates}, key=abs)

    def _grasp_plan(self) -> tuple[float, float, float]:
        """(TCP height above the body origin, yaw, per-finger pre-shape) of the first grasp, in order of
        preference (deep before shallow, wide margin before narrow, small yaw before large), whose hand is
        collision-free at the grasp pose with the fingers pre-shaped and closed onto the body, and 3 cm
        above it; the most preferred grasp if none is."""
        options = [(dz, yaw, self._preshape(yaw, margin)) for dz in self._grasp_offsets()
                   for margin in PRESHAPE_MARGINS for yaw in self._grasp_candidates()]
        for dz, yaw, finger in options:
            tcp = self._obj() + np.array([0.0, 0.0, dz])
            closed = self._preshape(yaw, 0.0005)  # pads at the body: nothing may lie between finger and body
            if (self._hand_free(tcp, yaw, finger) and self._hand_free(tcp, yaw, closed)
                    and self._hand_free(tcp + np.array([0.0, 0.0, 0.03]), yaw, finger)):
                return dz, yaw, finger
        return options[0]

    def _place_yaw(self, tcp_goal: np.ndarray, carry_z: float) -> float:
        """Carry yaw for the place: the grasp yaw unless the hand would hit a fixture on the vertical
        descent from carry height to the goal, else the smallest rotation (45 deg steps) that clears it.
        Success does not constrain the body's yaw."""
        yaw, finger = self.env.tcp_yaw, self.env.gripper_width
        column = [np.array([*tcp_goal[:2], z]) for z in np.linspace(tcp_goal[2], carry_z, 5)]
        for k in (0, 1, -1, 2, -2):
            c = yaw + k * np.pi / 4.0  # rotate the body by at most 90 deg; the closing axis repeats every 180 deg
            if abs(c) > YAW_LIMIT:
                c -= np.sign(c) * np.pi
            if all(self._hand_free(p, c, finger) for p in column):
                return c
        return yaw

    def _half_width(self, z_lo: float, z_hi: float) -> float:
        """Largest horizontal reach from the body origin of the body's collision geoms within the height
        band [z_lo, z_hi]: exact horizontal cross-sections for upright cylinders, spheres, and ellipsoids,
        bounding boxes otherwise."""
        env, m, d = self.env, self.env.model, self.env.data
        origin = self._obj()
        width = 0.0
        for g in env.object_geom_ids:
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            center = d.geom_xpos[g]
            size, kind = m.geom_size[g], m.geom_type[g]
            offset = float(np.linalg.norm(center[:2] - origin[:2]))
            upright = rot[2, 2] > 0.995
            if upright and kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                if center[2] + size[1] < z_lo or center[2] - size[1] > z_hi:
                    continue
                reach = size[0]
            elif upright and kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                a, c = (size[0], size[0]) if kind == mujoco.mjtGeom.mjGEOM_SPHERE else (max(size[0], size[1]), size[2])
                if center[2] + c < z_lo or center[2] - c > z_hi:
                    continue
                dz = 0.0 if z_lo <= center[2] <= z_hi else min(abs(z_lo - center[2]), abs(z_hi - center[2]))
                reach = a * np.sqrt(max(0.0, 1.0 - (dz / c) ** 2))
            else:
                box, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                corners = box + half * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                world = center + corners @ rot.T
                if world[:, 2].max() < z_lo or world[:, 2].min() > z_hi:
                    continue
                width = max(width, float(np.abs(world[:, :2] - origin[:2]).max()))
                continue
            width = max(width, float(reach + offset))
        return width

    def _grasp_offsets(self) -> list[float]:
        """Candidate TCP heights above the body origin for the grasp, deepest first (pads nearest the center
        of mass, least in-hand pivoting). A height qualifies if the palm clears the body top, the fingertips
        clear the support, the pads overlap the body by at least 5 mm, and nothing above the pads is wider
        than FINGER_INSET less than the body at the pads: the fingers' upper links reach that much further
        inward than the pads, so a wider rim or flange above the pads would stop the fingers first."""
        name = self.env.object_name
        if name in GRASP_HEIGHT_OVERRIDE:
            return [GRASP_HEIGHT_OVERRIDE[name]]
        lo, hi = self.env.object_extent_z()
        z = self._obj()[2]
        lowest = max(hi - PALM_ABOVE_TOP, lo + PAD_BELOW_TCP + 0.0025)
        heights = []
        for tcp_z in np.arange(lowest, hi - 0.002, 0.002):
            band_lo, band_hi = tcp_z - PAD_BELOW_TCP, tcp_z + PAD_ABOVE_TCP
            if min(band_hi, hi) - max(band_lo, lo) < 0.005:
                continue
            above = self._half_width(band_hi, hi) if band_hi < hi else 0.0
            if above <= self._half_width(band_lo, band_hi) - FINGER_INSET:
                heights.append(tcp_z - z)
        return heights or [max(hi - 0.012, lo + PAD_BELOW_TCP + 0.0025) - z]

    # ----------------------------------------------------------------- Phases
    def _press(self) -> None:
        goal = self.env.press_goal
        self.servo(goal + np.array([0.0, 0.0, 0.06]), 0.0, CLOSE, tol=0.005, max_steps=120)
        self.servo(goal, 0.0, CLOSE, tol=0.0005, max_steps=120, max_step=0.01)
        while True:
            self.act(self.action_towards(goal, 0.0, CLOSE, max_step=0.005))

    def _pick(self) -> bool:
        env = self.env
        dz, yaw, finger = self._grasp_plan()
        grip = _grip_command(finger)
        self.servo(lambda: self._obj() + np.array([0.0, 0.0, dz + 0.10]), yaw, grip, tol=0.004, max_steps=120)
        self.servo(lambda: self._obj() + np.array([0.0, 0.0, dz]), yaw, grip, tol=0.0015, max_steps=80, max_step=0.012)
        grasp_tcp = env.tcp_position.copy()  # hold the grasp pose against the reaction of the closing fingers
        for _ in range(12):
            self.act(self.action_towards(grasp_tcp, yaw, CLOSE))
            if env.is_grasped(env.target_id) and abs(env.joint_velocities).max() < 0.05:
                break
        return env.is_grasped(env.target_id)

    def _place(self) -> None:
        env = self.env
        goal = env.goal_position
        offset = env.tcp_position - self._obj()
        carry_z = max(env.tcp_position[2], goal[2] + offset[2]) + CARRY_CLEARANCE
        carry_z = max(carry_z, env.start_z + env.min_lift + offset[2] + 0.01)  # meet the task's lift requirement
        yaw = self._place_yaw(goal + offset, carry_z)
        self.servo(np.array([*env.tcp_position[:2], carry_z]), env.tcp_yaw, CLOSE, tol=0.005, max_steps=60)
        misses = [0]

        def lost() -> bool:
            misses[0] = 0 if env.is_grasped(env.target_id) else misses[0] + 1
            if misses[0] >= 3:
                raise GraspLost
            return False

        def tcp_for(obj_target: np.ndarray) -> np.ndarray:
            return env.tcp_position + (obj_target - self._obj())

        above = np.array([goal[0], goal[1], goal[2] + 0.05])
        self.servo(lambda: np.array([*tcp_for(above)[:2], carry_z]), yaw, CLOSE, tol=0.003, max_steps=120, until=lost)
        self.servo(lambda: tcp_for(above), yaw, CLOSE, tol=0.0008, max_steps=80, max_step=0.01, until=lost)
        if env.hold_goal:
            self.servo(lambda: tcp_for(goal), yaw, CLOSE, tol=0.0005, max_steps=120, max_step=0.004)
            while True:
                self.act(self.action_towards(tcp_for(goal), yaw, CLOSE, max_step=0.002))

        def supported() -> bool:
            contacts = env.body_contacts(env.target_id)
            return bool(contacts - env.finger_body_ids - env.robot_body_ids) and self._obj()[2] < goal[2] + 0.004

        for attempt in range(JAM_RETRIES + 1):
            if self._descend(goal, yaw, tcp_for, supported) or attempt == JAM_RETRIES:
                break
            # Stalled in contact above the goal (a part wedged at a tight fit): back off, re-centre, retry.
            entry = goal + np.array([0.0, 0.0, 0.012])
            self.servo(lambda: tcp_for(entry), yaw, CLOSE, tol=0.0005, max_steps=40, max_step=0.004, until=lost)
        self._release(yaw)

    def _release(self, yaw: float) -> None:
        """Open until neither finger touches the body, then back off slowly before retracting, so friction on
        a finger that still rests against the body does not drag it up."""
        env = self.env
        release_tcp = env.tcp_position.copy()
        for _ in range(20):
            self.act(self.action_towards(release_tcp, yaw, OPEN))
            if not env.body_contacts(env.target_id) & env.finger_body_ids and env.gripper_width > 0.5 * GRIPPER_OPEN_WIDTH:
                break
        self.servo(release_tcp + np.array([0.0, 0.0, 0.02]), yaw, OPEN, tol=0.003, max_steps=20, max_step=0.004)
        self.servo(env.tcp_position + np.array([0.0, 0.0, 0.05]), yaw, OPEN, tol=0.005, max_steps=30, max_step=0.01)

    def _descend(self, goal: np.ndarray, yaw: float, tcp_for, supported) -> bool:
        """Lower the body onto the goal; True once supported there, False if it stalls in contact above it."""
        env = self.env
        target = goal + np.array([0.0, 0.0, -0.002])
        history: list[float] = []
        for _ in range(120):
            if supported():
                return True
            z = self._obj()[2]
            history.append(z)
            in_contact = bool(env.body_contacts(env.target_id) - env.finger_body_ids - env.robot_body_ids)
            if in_contact and len(history) > 6 and history[-7] - z < 0.0005:
                return False
            step = 0.002 if z < goal[2] + 0.02 else 0.004
            self.act(self.action_towards(tcp_for(target), yaw, CLOSE, max_step=step))
        return False
