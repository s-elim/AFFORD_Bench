"""AFFORD-Bench task specification.

A task is an instruction, a start configuration, and a conjunction of goal predicates over simulated bodies and
mechanisms (as in LIBERO's BDDL goals). The episode succeeds at the first step at which every goal holds and every
ordering constraint is met, so it cannot end before the whole instruction is done.

Goals
- On(body, target):        body rests (released, supported, at rest, upright) at a scene target or on another free
                           body (stacking, sliding a gear onto a shaft), after having been lifted `min_lift`; region
                           targets (bins, trays, marked squares) accept any pose with the body inside the region.
                           `once` makes it a waypoint ("put the mug on the drip tray, brew, then serve it").
- Lifted(body, height):    body is held in a two-finger grasp at least `height` above its start, upright.
- Pressed(button):         a latching push button is pushed in (its joint is past the latch depth).
- Closed(slider) / Opened: a drawer, jaw, or lid slider is at its closed (open) end.
- GripperAbove(target, h): the fingertip center stays within `tol` of the point `h` above a target for `dwell` steps.
- Touching(body):          a fingertip touches the body's top for `dwell` steps without moving it more than 1 cm.
- HeldOver(body, target, c): body is held in a two-finger grasp with its lowest point `c` above the target's top
                           (a pipette tip over a well); `once` makes it a waypoint of a longer instruction.
- Slid(body, target):      body is slid along its support to a target without being lifted more than `max_lift`
                           (wiping with a sponge, sliding a block); it must be at rest there.

Starts
- Region(center, half):    uniform in a rectangle on the support below it, yaw within +/- yaw_deg.
- Slots(targets):          one of several receptacles (rack slots, compartments), chosen per episode.
- Placed(target):          resting at a target or on a free body (a pre-assembled part), the same every episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union


@dataclass(frozen=True)
class On:
    body: str
    target: str
    tol_xy: float | None = None
    tol_z: float | None = None
    tilt_deg: float | None = None
    min_lift: float = 0.02
    # A waypoint: satisfied once it has held at some step (the body moves on to a later goal).
    once: bool = False


@dataclass(frozen=True)
class Lifted:
    body: str
    height: float
    tilt_deg: float = 10.0


@dataclass(frozen=True)
class Pressed:
    button: str


@dataclass(frozen=True)
class Closed:
    slider: str


@dataclass(frozen=True)
class Opened:
    slider: str


@dataclass(frozen=True)
class GripperAbove:
    target: str
    height: float
    tol: float = 0.01
    dwell: int = 5


@dataclass(frozen=True)
class Touching:
    body: str
    dwell: int = 3
    # An event ("insert the pin, then press down on it"): satisfied once the touch has lasted `dwell` steps, so that
    # it can be combined with goals that need the fingers released afterwards.
    once: bool = False


@dataclass(frozen=True)
class HeldOver:
    body: str
    target: str
    clearance: float          # height of the body's lowest point above the target's top surface (m)
    tol: float = 0.01         # horizontal and vertical tolerance (m)
    once: bool = False


@dataclass(frozen=True)
class Slid:
    body: str
    target: str
    max_lift: float = 0.01
    tol_xy: float = 0.02


Goal = Union[On, Lifted, Pressed, Closed, Opened, GripperAbove, Touching, HeldOver, Slid]
# Goals the robot must keep satisfying until the episode ends; they can only be the last goal of a task (a HeldOver
# or Touching marked `once` is a waypoint or an event instead).
TERMINAL_GOALS = (Lifted, GripperAbove, Touching, HeldOver)


@dataclass(frozen=True)
class Region:
    center: tuple[float, float]
    half: tuple[float, float]
    yaw_deg: float = 45.0


@dataclass(frozen=True)
class Slots:
    targets: tuple[str, ...]
    yaw_deg: float = 45.0


@dataclass(frozen=True)
class Placed:
    target: str
    yaw_deg: float = 0.0


Start = Union[Region, Slots, Placed]


@dataclass(frozen=True)
class TaskSpec:
    number: int
    task_id: str
    domain: str
    scene: str
    instruction: str
    goals: tuple[Goal, ...]
    # Start overrides for free bodies (others start at their scene home).
    starts: Mapping[str, Start] = field(default_factory=dict)
    # Initial slider positions (m from the closed end) overriding the scene default.
    sliders: Mapping[str, float] = field(default_factory=dict)
    # (i, j): goal i first holds no later than goal j first holds ("put the mug on the tray, then press brew").
    order: tuple[tuple[int, int], ...] = ()
    max_steps: int | None = None
    # Free bodies in the scene besides those the goals and starts name; None takes the scene's first two distractors.
    extra_bodies: tuple[str, ...] | None = None
    # Provenance: the legacy task this one replaces (same number), its instruction, and why the definition changed
    # ("" if only the wording was tightened).
    legacy_task_id: str = ""
    legacy_instruction: str = ""
    change: str = ""

    def bodies_moved(self) -> tuple[str, ...]:
        """Free bodies the robot must move, in goal order."""
        out: list[str] = []
        for g in self.goals:
            body = getattr(g, "body", None)
            if body is not None and not isinstance(g, Touching) and body not in out:
                out.append(body)
        return tuple(out)

    def skills(self) -> int:
        """Number of manipulation primitives the instruction requires (sets the default step budget)."""
        return len(self.goals)


def step_budget(spec: TaskSpec) -> int:
    """Episode step limit (20 Hz): 400 steps for one primitive plus 250 for each further one, unless set. The oracle
    needs a median of 116 steps per primitive (task validation), so the budget leaves room for recoveries."""
    return spec.max_steps if spec.max_steps is not None else 400 + 250 * (spec.skills() - 1)
