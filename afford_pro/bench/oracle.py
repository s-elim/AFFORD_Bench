"""Privileged scripted expert for AFFORD-Bench: achieves the goals of a task in order.

Each goal maps to a primitive: On -> the grasp-carry-place routine of ParametricOracle on the focused body (with up to three attempts),
Lifted -> grasp and lift, Pressed -> push a button cap down past its latch, Closed -> push a slider face along its
closing direction, GripperAbove -> hover, Touching -> lower the closed fingers onto the body's top, Slid -> grasp the
body and drag it along its support. Goals that hold already are skipped.
"""
from __future__ import annotations

import numpy as np

from afford_pro.oracles.parametric_oracle import (CLOSE, GRASP_HEIGHT_OVERRIDE, GRASP_SYMMETRY, OPEN,
                                                  PRESHAPE_MARGINS, RECENTER_STEPS, YAW_LIMIT, GraspLost,
                                                  ParametricOracle, _grip_command, _wrap_half_turn)
from afford_pro.bench.env import GoalFrankaEnv
from afford_pro.oracles.parametric_oracle import CARRY_MAX_STEP
from afford_pro.bench.spec import Closed, GripperAbove, HeldOver, Lifted, On, Pressed, Slid, Touching  # noqa: F401

PLACE_ATTEMPTS = 3
APPROACH_HEIGHT = 0.06     # hover height above a button cap or a body top before descending (m)
BUTTON_OVERTRAVEL = 0.002  # push this far past the latch depth (m)
PUSH_STANDOFF = 0.03       # start a slider push this far behind the face (m)
PUSH_STEP = 0.006          # TCP step while pushing a slider (m per control step)
DRAG_STEP = 0.008          # TCP step while dragging a tool along its support (m per control step)
CLOSE_STEPS = 25           # longest wait for the fingers to close on a part (ParametricOracle: 12)
SEAT_SLACK = 0.0015        # a part that stops this close above its seat is released there and drops the rest (m)
STALL_STEPS = 6            # steps without descending that count as a stall
DESCEND_STEPS = 120
TRAVEL_CLEARANCE = 0.12    # highest empty-hand crossing above the approach point; higher, the arm stretched (m)
SETTLE_STEPS = 20          # longest wait for the arm to stop above a point before descending onto it
SETTLED_JOINT_SPEED = 0.05 # joint speed (rad/s) below which the arm counts as stopped


class GoalOracle(ParametricOracle):
    def __init__(self, env: GoalFrankaEnv):
        super().__init__(env)
        self.max_steps = env.max_episode_steps

    def run(self) -> None:
        for index, goal in enumerate(self.env.task_spec.goals):
            if self.env.goal_holds(index):
                continue
            self._achieve(index, goal)
        while True:  # wait for bodies to settle; the episode ends on success or at the step budget
            self.hold(OPEN, 1)

    def _achieve(self, index: int, goal) -> None:
        if isinstance(goal, On):
            self._place_goal(index)
        elif isinstance(goal, Lifted):
            self._lift(index, goal)
        elif isinstance(goal, Pressed):
            self._press_button(goal.button)
        elif isinstance(goal, Closed):
            self._push_slider(goal.slider)
        elif isinstance(goal, GripperAbove):
            self._hover(goal)
        elif isinstance(goal, Touching):
            self._touch(index, goal)
        elif isinstance(goal, HeldOver):
            self._hold_over(index, goal)
        elif isinstance(goal, Slid):
            self._slide_along(goal)
        else:
            raise NotImplementedError(f"no primitive for {type(goal).__name__}")

    # ----------------------------------------------------------------- Grasp hints from the scene
    def _grasp_candidates(self) -> list[float]:
        env = self.env
        hint = env.scene.bodies[env.object_name].grasp_symmetry
        sym = hint if hint is not None else GRASP_SYMMETRY.get(env.object_name)
        if sym is None:
            candidates = [k * np.pi / 4.0 for k in (-1, 0, 1, 2)]
        else:
            rot = env.data.xmat[env.target_id].reshape(3, 3)
            base = float(np.arctan2(rot[1, 0], rot[0, 0]))
            candidates = [base + k * sym for k in range(max(1, int(round(np.pi / sym))))]
        return sorted({round(_wrap_half_turn(c), 9) for c in candidates}, key=abs)

    def _grasp_offsets(self) -> list[float]:
        hint = self.env.scene.bodies[self.env.object_name].grasp_height
        if hint is not None:
            return [hint]
        if self.env.object_name in GRASP_HEIGHT_OVERRIDE:
            return [GRASP_HEIGHT_OVERRIDE[self.env.object_name]]
        return super()._grasp_offsets()

    def grasp_exists(self) -> bool:
        """Whether one of the grasps this oracle would try on the focused body is collision-free (used to accept
        start layouts: a single-height check rejected parts the oracle grasps above low walls)."""
        for dz in self._grasp_offsets():
            for margin in PRESHAPE_MARGINS:
                for yaw in self._grasp_candidates():
                    tcp = self._obj() + np.array([0.0, 0.0, dz])
                    finger, closed = self._preshape(yaw, margin), self._preshape(yaw, 0.0005)
                    if (self._hand_free(tcp, yaw, finger) and self._hand_free(tcp, yaw, closed)
                            and self._hand_free(tcp + np.array([0.0, 0.0, 0.03]), yaw, finger)):
                        return True
        return False

    def _place_yaw(self, tcp_goal: np.ndarray, carry_z: float) -> float:
        """For a part that must be turned to fit its socket, the hand yaw that turns the part to the socket's yaw
        (the smallest such turn whose descent is collision-free); otherwise ParametricOracle's collision-avoiding yaw."""
        env = self.env
        sym = env.scene.bodies[env.object_name].yaw_symmetry
        goal = None if env.focused_goal is None else env.task_spec.goals[env.focused_goal]
        target_yaw = env.goal_yaw(goal) if isinstance(goal, On) else None
        if sym is None or target_yaw is None:
            return super()._place_yaw(tcp_goal, carry_z)
        turn = (target_yaw - env.body_yaw(env.object_name) + sym / 2.0) % sym - sym / 2.0
        hand = env.tcp_yaw
        options = sorted((hand + turn + k * sym for k in range(-4, 5)), key=lambda c: abs(c - hand))
        options = [c for c in options if abs(c) <= YAW_LIMIT] or [hand + turn]
        column = [np.array([*tcp_goal[:2], z]) for z in np.linspace(tcp_goal[2], carry_z, 5)]
        finger = env.gripper_width
        for c in options:
            if all(self._hand_free(p, c, finger) for p in column):
                return c
        return options[0]

    # ----------------------------------------------------------------- Primitives
    def _pick(self) -> bool:
        """ParametricOracle's pick with a longer closing window: fingers pre-shaped for a wide body (a lid) need more than its
        12 steps to close on a narrow part of it (the knob)."""
        env = self.env
        dz, yaw, finger = self._grasp_plan()
        grip = _grip_command(finger)
        self.servo(lambda: self._obj() + np.array([0.0, 0.0, dz + 0.10]), yaw, grip, tol=0.004, max_steps=120)
        self.servo(lambda: self._obj() + np.array([0.0, 0.0, dz]), yaw, grip, tol=0.0015, max_steps=80, max_step=0.012)
        grasp_tcp = env.tcp_position.copy()
        for _ in range(CLOSE_STEPS):
            self.act(self.action_towards(grasp_tcp, yaw, CLOSE))
            if env.is_grasped(env.target_id) and abs(env.joint_velocities).max() < 0.05:
                break
        return env.is_grasped(env.target_id)

    def _descend(self, goal: np.ndarray, yaw: float, tcp_for, supported) -> bool:
        """Lower the body onto the goal; True once it is supported there or has stopped within SEAT_SLACK above it
        (the fingertips resting on a socket's rim), False if it stalls higher (a jam to back off from)."""
        env = self.env
        target = goal + np.array([0.0, 0.0, -0.002])
        history: list[float] = []
        recenter_budget = RECENTER_STEPS
        for _ in range(DESCEND_STEPS):
            if supported():
                return True
            obj = self._obj()
            history.append(float(obj[2]))
            if len(history) > STALL_STEPS and history[-STALL_STEPS - 1] - obj[2] < 0.0005:
                return bool(obj[2] - goal[2] <= SEAT_SLACK and np.linalg.norm(obj[:2] - goal[:2]) <= env.tolerance_xy)
            step = 0.002 if obj[2] < goal[2] + 0.02 else 0.004
            off_center = np.linalg.norm(obj[:2] - goal[:2]) > 0.001
            recenter = off_center and obj[2] < goal[2] + 0.015 and recenter_budget > 0
            recenter_budget -= int(recenter)
            aim = np.array([goal[0], goal[1], obj[2]]) if recenter else target
            self.act(self.action_towards(tcp_for(aim), yaw, CLOSE, max_step=step))
        return False

    def _travel(self, point: np.ndarray, grip: float, tol: float = 0.004) -> None:
        """Empty-hand move to `point`: across at the hand's height (raised to the point's height if lower, brought down
        to TRAVEL_CLEARANCE above it if higher), then straight down once the arm has stopped, so the fingers never
        descend diagonally next to what they pass (one pushed a seated lid off its seat). Anything taller than the
        crossing height is not avoided."""
        env = self.env
        z = max(point[2], min(env.tcp_position[2], point[2] + TRAVEL_CLEARANCE))
        if env.tcp_position[2] < z:
            self.servo(np.array([*env.tcp_position[:2], z]), 0.0, grip, tol=tol, max_steps=40)
        cross = np.array([point[0], point[1], z])
        self.servo(cross, 0.0, grip, tol=tol, max_steps=150)
        for _ in range(SETTLE_STEPS):  # let the arm stop above the point, or it overshoots sideways while descending
            if np.abs(env.joint_velocities).max() < SETTLED_JOINT_SPEED:
                break
            self.act(self.action_towards(cross, 0.0, grip))
        self.servo(point, 0.0, grip, tol=tol, max_steps=60)

    def _retreat(self) -> None:
        self.servo(self.env.tcp_position + np.array([0.0, 0.0, APPROACH_HEIGHT]), None, OPEN, max_steps=15)

    def _holding(self, body: str) -> bool:
        return self.env.is_grasped(self.env.body_ids[body])

    def _grasp(self, body: str) -> bool:
        """Grasp `body` unless it is already in the hand (up to PLACE_ATTEMPTS tries)."""
        for _ in range(PLACE_ATTEMPTS):
            self.env.focus(body)
            if self._holding(body) or self._pick():
                return True
            self._retreat()
        return False

    def _place_goal(self, index: int) -> None:
        env = self.env
        for _ in range(PLACE_ATTEMPTS):
            env.focus_goal(index)
            try:
                if self._holding(env.object_name) or self._pick():
                    self._place()
            except GraspLost:
                pass
            if env.on_near(env.task_spec.goals[index]):
                return
            self._retreat()

    def _lift(self, index: int, goal: Lifted) -> None:
        env = self.env
        if not self._grasp(goal.body):
            return
        top = env.tcp_position + np.array([0.0, 0.0, goal.height + 0.01])
        self.servo(top, env.tcp_yaw, CLOSE, tol=0.003, max_steps=120, max_step=0.012)
        while True:
            self.act(self.action_towards(top, env.tcp_yaw, CLOSE, max_step=0.002))

    def _press_button(self, name: str) -> None:
        env = self.env
        cap = env.button_cap[name]
        xy = env.data.xpos[cap][:2].copy()
        top = env.body_top_of_id(cap)
        above = np.array([xy[0], xy[1], top + env.PAD_BELOW_TCP + APPROACH_HEIGHT])
        self._travel(above, CLOSE)
        latch = env.scene.buttons[name].latch
        push = np.array([xy[0], xy[1], top + env.PAD_BELOW_TCP - latch - BUTTON_OVERTRAVEL])
        for _ in range(80):
            if env.button_depth(name) >= latch + 0.5 * BUTTON_OVERTRAVEL:
                break
            self.act(self.action_towards(push, 0.0, CLOSE, max_step=0.004))
        self.servo(above, 0.0, CLOSE, tol=0.005, max_steps=40)

    def _hold_over(self, index: int, goal: HeldOver) -> None:
        """Grasp the body, carry it over the target and lower it until its lowest point is `clearance` above the
        target's top; hold there, or move on once a waypoint has been reached."""
        env = self.env
        if not self._grasp(goal.body):
            return
        env.focus(goal.body)
        bid = env.body_ids[goal.body]

        def tcp_target() -> np.ndarray:
            point = env.target_point(goal.target)
            origin = env.data.xpos[bid]
            want = np.array([point[0], point[1], env.surface_top(goal.target) + goal.clearance
                             + origin[2] - env.body_bottom(goal.body)])
            return env.tcp_position + (want - origin)

        carry_z = max(env.tcp_position[2] + 0.05, self._path_clear_z(env.target_point(goal.target)))
        yaw = env.tcp_yaw
        self.servo(np.array([*env.tcp_position[:2], carry_z]), yaw, CLOSE, tol=0.005, max_steps=60,
                   max_step=CARRY_MAX_STEP)
        self.servo(lambda: np.array([*tcp_target()[:2], carry_z]), yaw, CLOSE, tol=0.003, max_steps=160,
                   max_step=CARRY_MAX_STEP)
        self.servo(tcp_target, yaw, CLOSE, tol=0.002, max_steps=80, max_step=0.006)
        while not (goal.once and env.goal_holds(index)):
            self.act(self.action_towards(tcp_target(), yaw, CLOSE, max_step=0.002))

    def _push_slider(self, name: str) -> None:
        env = self.env
        slider = env.scene.sliders[name]
        direction = np.asarray(slider.push_dir, dtype=np.float64)
        face = env.slider_face_point(name)
        start = face - direction * PUSH_STANDOFF + np.array([0.0, 0.0, env.PAD_BELOW_TCP])
        self._travel(start + np.array([0.0, 0.0, APPROACH_HEIGHT]), CLOSE)
        self.servo(start, 0.0, CLOSE, tol=0.002, max_steps=60, max_step=0.008)
        end = start + direction * (PUSH_STANDOFF + env.slider_position(name) + 0.002)
        for _ in range(120):
            if env.slider_position(name) <= 0.5 * slider.tol:
                break
            self.act(self.action_towards(end, 0.0, CLOSE, max_step=PUSH_STEP))
        back = env.tcp_position - direction * PUSH_STANDOFF
        self.servo(back, 0.0, CLOSE, tol=0.004, max_steps=30)
        self.servo(back + np.array([0.0, 0.0, APPROACH_HEIGHT]), 0.0, CLOSE, tol=0.005, max_steps=30)

    def _hover(self, goal: GripperAbove) -> None:
        point = self.env.target_point(goal.target) + np.array([0.0, 0.0, goal.height])
        self._travel(point, OPEN, tol=0.002)
        while True:
            self.act(self.action_towards(point, 0.0, OPEN, max_step=0.002))

    def _touch(self, index: int, goal: Touching) -> None:
        """Lower the closed fingers onto the body's top and stay there (or, for a touch event, lift off once it has
        lasted long enough)."""
        env = self.env
        bid = env.body_ids[goal.body]
        xy = env.data.xpos[bid][:2].copy()
        top = env.body_top(goal.body)
        above = np.array([xy[0], xy[1], top + env.PAD_BELOW_TCP + APPROACH_HEIGHT])
        self._travel(above, CLOSE)
        target = np.array([xy[0], xy[1], top + env.PAD_BELOW_TCP - 0.002])
        while not (goal.once and env.goal_holds(index)):
            step = 0.002 if env.body_contacts(bid) & env.finger_body_ids else 0.006
            self.act(self.action_towards(target, 0.0, CLOSE, max_step=step))
        self.servo(above, 0.0, CLOSE, tol=0.005, max_steps=40, max_step=0.01)

    def _slide_along(self, goal: Slid) -> None:
        env = self.env
        if not self._grasp(goal.body):
            return
        tool = env.data.xpos[env.body_ids[goal.body]]
        offset = env.tcp_position - tool
        goal_xy = env.target_point(goal.target)[:2]
        end = np.array([goal_xy[0] + offset[0], goal_xy[1] + offset[1], env.tcp_position[2] - 0.002])
        yaw = env.tcp_yaw
        self.servo(end, yaw, CLOSE, tol=0.003, max_steps=200, max_step=DRAG_STEP)
        self._release(yaw)
