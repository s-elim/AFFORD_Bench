"""Closed-loop oracle for the Family C twist-lock mechanisms (Tasks 227-233).

Push-and-turn bodies (bayonet cartridge, safety cap) are grasped on their axis, pressed down until the axial slide
reaches the required compression (closed loop on the slide position, so the spring preload is held while turning),
turned under that preload until the mechanism angle reaches its target, then unloaded to the seat or top position,
released, and lifted. A turn larger than 120 deg is split into strokes: turn, release, counter-rotate the open
hand, regrasp, press again, and complete the turn. The key is grasped by its bow, carried over the keyway, aligned
with the sleeve slot, inserted to the plug floor, turned 90 deg about the keyway axis, and released; for Task 231
the bolt is then grasped by its knob and slid open.
"""
from __future__ import annotations

import mujoco
import numpy as np

from afford_pro.oracles.frontier_oracle import FrontierOracle
from afford_pro.oracles.parametric_oracle import CLOSE, OPEN, _grip_command

PRESS_DEPTH = 0.0125          # target axial compression (m); the slides stop at 13 mm (cap) and 14 mm (bayonet)
REGRASP_THRESHOLD = np.radians(120.0)  # larger turns are split into strokes with a regrasp in between
MAX_STROKES = 4
STROKE_MARGIN = np.radians(30.0)  # hand travel beyond the planned mechanism travel (tooth backlash, slip)
TURN_RATE = 0.035             # rad per 50 ms step while turning (0.7 rad/s)
ANGLE_DONE = np.radians(0.3)  # stop turning within this of the target angle
PRESS_STEP = 0.004            # largest axial TCP correction per step (m)
CAP_TOOTH_PITCH = np.radians(30.0)  # ring tooth pitch of the safety cap; its six teeth sit mid-gap at zero offset
PRESS_LEAD = 0.003            # largest integral lead of the TCP goal beyond the grasped body (m)
GRASP_BELOW_TOP = 0.012       # TCP below the top of a turned body (m)
KEY_GRASP_BELOW_TOP = 0.009   # TCP below the top of the key bow (m)
KEY_DEPTH = 0.024             # insertion target below the keyway top (m); the plug floor stops the blade at 26 mm
KEY_ALIGN_XY = 0.0004         # blade-to-axis alignment before inserting (m)
CARRY_Z = 0.16                # TCP height for carrying the key (m)
BOLT_TARGET = 0.034           # bolt opening commanded for Task 231 (m)
CARRY_MAX_STEP = 0.012        # m per step while carrying the key (as the parametric oracle's carry legs)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class TwistLockOracle(FrontierOracle):
    """Privileged expert for TwistLockEnv; `press_depth` overrides the axial compression (used to show that an
    under-threshold press cannot turn the cap)."""

    max_steps = 900

    def __init__(self, env, press_depth: float | None = None):
        super().__init__(env)
        self.press_depth = PRESS_DEPTH if press_depth is None else press_depth
        self._press_i = 0.0
        self._grasp_offset_z = 0.0

    def run(self) -> None:
        env = self.env
        if env.mech.kind == "key":
            for _ in range(3):
                if self._key_insert_and_turn():
                    break
            if env.mech.min_bolt_travel > 0.0:
                self._slide_bolt()
        else:
            for angle, press in env.mech.stages:
                self._push_turn(angle, 0.0 if press is None else press)
        while True:  # the episode ends on success or at the step budget
            self.hold(OPEN, 1)

    # ----------------------------------------------------------------- Low-level actions
    def _step(self, delta_xyz: np.ndarray, yaw: float | None, grip: float, yaw_rate: float | None = None) -> None:
        env = self.env
        a = np.zeros(7)
        a[:3] = np.clip(np.asarray(delta_xyz) / env.action_scale, -1.0, 1.0)
        if yaw is not None:
            dyaw = _wrap(yaw - env.tcp_yaw)
            if yaw_rate is not None:
                dyaw = float(np.clip(dyaw, -yaw_rate, yaw_rate))
            a[5] = np.clip(dyaw / env.yaw_scale, -1.0, 1.0)
        a[6] = grip
        self._prev_delta = np.zeros(3)
        self.act(a)

    def _axis(self) -> np.ndarray:
        """World xy of the turned body's fixture axis (its parent fixture body)."""
        env = self.env
        return env.data.xpos[env.model.body_parentid[env.mech_body_id]][:2].copy()

    def _top(self) -> float:
        return self.env.object_extent_z()[1]

    def _close(self, yaw: float, steps: int = 12) -> None:
        hold = self.env.tcp_position.copy()
        for _ in range(steps):
            self._step(hold - self.env.tcp_position, yaw, CLOSE)

    def _open_and_lift(self, yaw: float | None, lift: float = 0.05) -> None:
        hold = self.env.tcp_position.copy()
        for _ in range(8):
            self._step(hold - self.env.tcp_position, yaw, OPEN)
        self.servo(hold + np.array([0.0, 0.0, lift]), yaw, OPEN, tol=0.004, max_steps=30)

    # ----------------------------------------------------------------- Push-and-turn (cap, bayonet)
    def _grasp_axis(self, yaw: float) -> None:
        env = self.env
        axis = self._axis()
        z = self._top() - GRASP_BELOW_TOP
        finger = self._preshape(yaw, 0.008)
        grip = _grip_command(finger)
        above = np.array([axis[0], axis[1], z + 0.06])
        if env.tcp_position[2] < z + 0.04:
            self.servo(env.tcp_position + np.array([0.0, 0.0, 0.05]), None, grip, tol=0.004, max_steps=20)
        self.servo(above, yaw, grip, tol=0.003, max_steps=120)
        self.servo(np.array([axis[0], axis[1], z]), yaw, grip, tol=0.0015, max_steps=60, max_step=0.008)
        self._close(yaw)
        self._press_i = 0.0
        self._grasp_offset_z = float(env.tcp_position[2] - env.data.xpos[env.mech_body_id][2])

    def _press_action(self, target: float) -> np.ndarray:
        """TCP displacement that drives the axial slide to `target` (m) while holding the TCP on the fixture axis.

        The TCP goal follows the grasped body (its rest height plus the target compression, at the grasp offset) with
        a bounded integral term for the arm's compliance; the bound keeps the commanded TCP within PRESS_LEAD of the
        body, so a blocked press (cap teeth landing on the ring teeth) does not build up a force that slips the grasp."""
        env = self.env
        err = target - env.press_position
        self._press_i = float(np.clip(self._press_i + 0.3 * err, -PRESS_LEAD, PRESS_LEAD))
        body_rest_z = float(env.data.xpos[env.mech_body_id][2]) - env.press_position
        goal_z = body_rest_z + target + self._grasp_offset_z + self._press_i
        dz = float(np.clip(goal_z - env.tcp_position[2], -PRESS_STEP, PRESS_STEP))
        dxy = self._axis() - env.tcp_position[:2]
        return np.array([dxy[0], dxy[1], dz])

    def _press_to(self, target: float, yaw: float, steps: int = 40, tol: float = 0.0005) -> None:
        for _ in range(steps):
            if abs(self.env.press_position - target) < tol:
                return
            self._step(self._press_action(target), yaw, CLOSE)

    def _align_cap_teeth(self, yaw: float) -> float:
        """Turn the unpressed (freewheeling) cap so that its teeth face the gaps between the ring teeth; pressed onto
        the tooth tops instead, the cap stops at dz = 10 mm and the arm's push can exceed the grasp friction."""
        env = self.env
        offset = float(env.data.qpos[env.spin_qadr]) - env.mechanism_angle  # cap teeth sit mid-gap at offset 0 mod pitch
        err = (offset + CAP_TOOTH_PITCH / 2.0) % CAP_TOOTH_PITCH - CAP_TOOTH_PITCH / 2.0
        goal = yaw - err
        for _ in range(20):
            if abs(_wrap(goal - env.tcp_yaw)) < np.radians(1.0):
                break
            self._step(self._press_action(0.0), goal, CLOSE, yaw_rate=TURN_RATE)
        return goal

    def _turn(self, target: float, yaw_end: float, press: float) -> None:
        """Turn the hand toward `yaw_end` under preload until the mechanism angle reaches `target`."""
        env = self.env
        direction = np.sign(yaw_end - env.tcp_yaw) or 1.0
        for _ in range(240):
            err = target - env.mechanism_angle
            if abs(err) < ANGLE_DONE:
                return
            if np.sign(err) != direction:  # overshot: hold here and let the stroke end
                return
            remaining_hand = (yaw_end - env.tcp_yaw) * direction
            if remaining_hand <= 0.0:
                return
            rate = float(np.clip(0.8 * abs(err), 0.006, TURN_RATE))
            self._step(self._press_action(press), env.tcp_yaw + direction * rate, CLOSE, yaw_rate=rate)

    def _push_turn(self, target: float, rest_press: float) -> None:
        env = self.env
        for _ in range(MAX_STROKES):
            remaining = target - env.mechanism_angle
            if abs(remaining) < ANGLE_DONE * 3:
                break
            direction = float(np.sign(remaining))
            travel = abs(remaining) / 2.0 if abs(remaining) > REGRASP_THRESHOLD else abs(remaining)
            stroke_target = env.mechanism_angle + direction * travel
            span = travel + STROKE_MARGIN
            yaw0 = -direction * span / 2.0
            self._grasp_axis(yaw0)
            if env.mech.kind == "cap":
                yaw0 = self._align_cap_teeth(yaw0)
            self._press_to(-self.press_depth, yaw0)
            self._turn(stroke_target, yaw0 + direction * span, -self.press_depth)
            # Unload with the grip closed (the body rises to its seat or top), then release and lift.
            self._press_to(rest_press, None, steps=40)
            self._open_and_lift(None)
        # Leave the hand clear while the terminal state is held.
        self.servo(self.env.tcp_position + np.array([0.0, 0.0, 0.04]), None, OPEN, tol=0.004, max_steps=15)
        if len(env.mech.stages) > 1:
            self.hold(OPEN, 14)  # let the intermediate seat register before the next stage

    # ----------------------------------------------------------------- Key, latch, bolt
    def _key_tip(self) -> np.ndarray:
        return self.env.data.xpos[self.env.mech_body_id].copy()

    def _key_yaw(self) -> float:
        rot = self.env.data.xmat[self.env.mech_body_id].reshape(3, 3)
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _lock_axis(self) -> np.ndarray:
        env = self.env
        return env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "lock_housing")][:2].copy()

    def _sleeve_angle(self) -> float:
        env = self.env
        j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "lock_sleeve_hinge")
        return float(env.data.qpos[env.model.jnt_qposadr[j]])

    def _key_insert_and_turn(self) -> bool:
        env = self.env
        key = env.mech_body_id
        if not env.is_grasped(key):
            tip = self._key_tip()
            yaw = _wrap(self._key_yaw())
            yaw = yaw - np.pi if yaw > np.pi / 2 else yaw + np.pi if yaw < -np.pi / 2 else yaw
            bow = tip + env.data.xmat[key].reshape(3, 3) @ np.array([0.0, 0.0, 0.040])
            z = self._top() - KEY_GRASP_BELOW_TOP
            finger = self._preshape(yaw, 0.008)
            grip = _grip_command(finger)
            self.servo(np.array([bow[0], bow[1], z + 0.06]), yaw, grip, tol=0.003, max_steps=120)
            self.servo(np.array([bow[0], bow[1], z]), yaw, grip, tol=0.0015, max_steps=60, max_step=0.008)
            self._close(yaw)
            if not env.is_grasped(key):
                self._open_and_lift(yaw)
                return False
        axis = self._lock_axis()
        top = env.keyway_top_z

        def aligned_target(tip_z: float) -> np.ndarray:
            tip = self._key_tip()
            tcp = env.tcp_position
            return np.array([tcp[0] + axis[0] - tip[0], tcp[1] + axis[1] - tip[1], tcp[2] + tip_z - tip[2]])

        def key_yaw_goal() -> float:
            # The blade must lie along the sleeve slot (either way round); turn the hand by the key's yaw error.
            err = _wrap(self._sleeve_angle() - self._key_yaw())
            err = err - np.pi if err > np.pi / 2 else err + np.pi if err < -np.pi / 2 else err
            return env.tcp_yaw + err

        # Lift, carry above the keyway, align, then insert slowly with continuous lateral correction.
        self.servo(np.array([*env.tcp_position[:2], CARRY_Z]), None, CLOSE, tol=0.004, max_steps=40)
        self.servo(lambda: aligned_target(top + 0.06), None, CLOSE, tol=0.003, max_steps=120,
                   max_step=CARRY_MAX_STEP)
        for _ in range(60):
            target = aligned_target(top + 0.003)
            yaw_goal = key_yaw_goal()
            if (np.linalg.norm(target - env.tcp_position) < KEY_ALIGN_XY + 0.0005
                    and abs(_wrap(yaw_goal - env.tcp_yaw)) < np.radians(1.0)):
                break
            self._step(np.clip(target - env.tcp_position, -0.006, 0.006), yaw_goal, CLOSE)
        stalled, last = 0, self._key_tip()[2]
        for _ in range(80):
            if top - self._key_tip()[2] >= KEY_DEPTH:
                break
            target = aligned_target(self._key_tip()[2] - 0.003)
            self._step(np.clip(target - env.tcp_position, -0.004, 0.004), key_yaw_goal(), CLOSE)
            tip_z = self._key_tip()[2]
            stalled = stalled + 1 if last - tip_z < 0.0003 else 0
            last = tip_z
            if stalled >= 8:
                break
        if top - self._key_tip()[2] < KEY_DEPTH - 0.002:
            self.servo(lambda: aligned_target(top + 0.01), None, CLOSE, tol=0.002, max_steps=30)
            return False
        # Turn about the keyway axis: the hand yaw advances while the TCP keeps the blade on the axis.
        target_angle = env.mech.stages[0][0]
        for _ in range(120):
            err = target_angle - env.mechanism_angle
            if err < np.radians(0.5):
                break
            rate = float(np.clip(0.8 * err, 0.01, TURN_RATE))
            tgt = aligned_target(self._key_tip()[2])
            self._step(np.clip(tgt - env.tcp_position, -0.004, 0.004), env.tcp_yaw + rate, CLOSE, yaw_rate=rate)
        self._open_and_lift(None, lift=0.06)
        return True

    def _slide_bolt(self) -> None:
        env = self.env
        m, d = env.model, env.data
        knob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lock_bolt_knob")
        bolt_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "lock_bolt_slide")]
        yaw = np.pi / 2.0  # closing axis along x: the pads push on the knob faces normal to the slide
        top = d.geom_xpos[knob][2] + m.geom_size[knob][2]
        grip = _grip_command(m.geom_size[knob][0] + 0.008)
        z = top - 0.008
        self.servo(np.array([*d.geom_xpos[knob][:2], z + 0.07]), yaw, grip, tol=0.003, max_steps=120)
        self.servo(lambda: np.array([*d.geom_xpos[knob][:2], z]), yaw, grip, tol=0.0015, max_steps=60, max_step=0.008)
        self._close(yaw)
        for _ in range(80):
            remaining = BOLT_TARGET - float(d.qpos[bolt_q])
            if remaining < 0.0005:
                break
            target = np.array([d.geom_xpos[knob][0] + min(remaining, 0.004), d.geom_xpos[knob][1], z])
            self._step(np.clip(target - env.tcp_position, -0.005, 0.005), yaw, CLOSE)
        self._open_and_lift(yaw, lift=0.06)

