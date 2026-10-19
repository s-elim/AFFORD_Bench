"""Goal predicates of AFFORD-Bench, evaluated on the current simulator state of a GoalFrankaEnv.

State predicates (On, Pressed, Closed, Opened, Lifted) are functions of the physics state, plus each body's start
height and highest point since reset (On's lift requirement, Slid's no-lift requirement). Dwell predicates
(GripperAbove, Touching) count consecutive steps; their counters are kept by the environment.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from afford_pro.bench.spec import Closed, GripperAbove, HeldOver, Lifted, On, Opened, Pressed, Slid, Touching

if TYPE_CHECKING:
    from afford_pro.bench.env import GoalFrankaEnv

REST_LIN_SPEED = 0.02    # m/s: a placed body counts as at rest below this speed
REST_ANG_SPEED = 0.2     # rad/s
TOUCH_MAX_SHIFT = 0.01   # a touched body may move at most this far from its start (m)
TOUCH_TOP_MARGIN = 0.005 # the fingertips must be no lower than this below the body's top (m)


def on_holds(env: GoalFrankaEnv, goal: On) -> bool:
    if not env.on_near(goal):
        return False
    if env.peak_z[goal.body] - env.start_z_of[goal.body] < goal.min_lift - 1e-6:
        return False
    bid = env.body_ids[goal.body]
    lin, ang = env.body_speed(goal.body)
    if lin >= REST_LIN_SPEED or ang >= REST_ANG_SPEED:
        return False
    contacts = env.body_contacts(bid)
    released = not (contacts & env.finger_body_ids)
    supported = bool(contacts - env.finger_body_ids - env.robot_body_ids)
    return released and supported


def lifted_holds(env: GoalFrankaEnv, goal: Lifted) -> bool:
    bid = env.body_ids[goal.body]
    high = env.data.xpos[bid][2] - env.start_z_of[goal.body] >= goal.height - 0.005
    upright = env.body_tilt(goal.body) <= np.radians(goal.tilt_deg)
    return bool(env.is_grasped(bid) and high and upright and env.body_speed(goal.body)[0] < 0.05)


def held_over_holds(env: GoalFrankaEnv, goal: HeldOver) -> bool:
    bid = env.body_ids[goal.body]
    if not env.is_grasped(bid):
        return False
    point = env.target_point(goal.target)
    gap = env.body_bottom(goal.body) - env.surface_top(goal.target)
    near = np.linalg.norm(env.data.xpos[bid][:2] - point[:2]) <= goal.tol
    return bool(near and abs(gap - goal.clearance) <= goal.tol and env.body_speed(goal.body)[0] < 0.05)


def gripper_above_now(env: GoalFrankaEnv, goal: GripperAbove) -> bool:
    point = env.target_point(goal.target) + np.array([0.0, 0.0, goal.height])
    return bool(np.linalg.norm(env.tcp_position - point) <= goal.tol)


def touching_now(env: GoalFrankaEnv, goal: Touching) -> bool:
    bid = env.body_ids[goal.body]
    if not env.body_contacts(bid) & env.finger_body_ids:
        return False
    anchor = env.active_pos.get(env.task_spec.goals.index(goal), env.start_pos_of[goal.body])
    shift = np.linalg.norm(env.data.xpos[bid] - anchor)
    top = env.body_top(goal.body)
    fingertips = env.tcp_position[2] - env.PAD_BELOW_TCP
    return bool(shift <= TOUCH_MAX_SHIFT and fingertips >= top - TOUCH_TOP_MARGIN)


def slid_holds(env: GoalFrankaEnv, goal: Slid) -> bool:
    bid = env.body_ids[goal.body]
    pos = env.data.xpos[bid]
    near = np.linalg.norm(pos[:2] - env.target_point(goal.target)[:2]) <= goal.tol_xy
    stayed_down = env.peak_z[goal.body] - env.start_z_of[goal.body] <= goal.max_lift
    return bool(near and stayed_down and env.body_speed(goal.body)[0] < REST_LIN_SPEED)


def state_holds(env: GoalFrankaEnv, goal) -> bool:
    """Truth of a goal that does not need a dwell counter."""
    if isinstance(goal, On):
        return on_holds(env, goal)
    if isinstance(goal, Lifted):
        return lifted_holds(env, goal)
    if isinstance(goal, HeldOver):
        return held_over_holds(env, goal)
    if isinstance(goal, Pressed):
        return env.button_depth(goal.button) >= env.scene.buttons[goal.button].latch
    if isinstance(goal, Closed):
        return env.slider_position(goal.slider) <= env.scene.sliders[goal.slider].tol
    if isinstance(goal, Opened):
        slider = env.scene.sliders[goal.slider]
        return env.slider_position(goal.slider) >= slider.open - slider.tol
    if isinstance(goal, Slid):
        return slid_holds(env, goal)
    raise TypeError(f"{type(goal).__name__} needs a dwell counter")


def dwell_condition(env: GoalFrankaEnv, goal) -> bool:
    """Per-step condition whose consecutive count makes a dwell goal true."""
    if isinstance(goal, GripperAbove):
        return gripper_above_now(env, goal)
    if isinstance(goal, Touching):
        return touching_now(env, goal)
    raise TypeError(f"{type(goal).__name__} is not a dwell goal")


DWELL_GOALS = (GripperAbove, Touching)
