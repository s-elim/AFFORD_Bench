"""Language for AFFORD-Bench: noun phrases for every scene entity and a step-by-step instruction derived from a
task's goals, so the paraphrase states exactly what the success check requires (a LIBERO-PRO-style instruction
variant). The canonical instruction stays the hand-written `TaskSpec.instruction`.
"""
from __future__ import annotations

from afford_pro.bench.spec import Closed, GripperAbove, HeldOver, Lifted, On, Opened, Pressed, Slid, TaskSpec, Touching

# Free bodies and mechanisms: noun phrase.
THINGS: dict[str, str] = {
    "machined_shaft": "the transmission shaft", "spur_gear": "the spur gear", "dowel_pin": "the dowel pin",
    "bearing_ring": "the bearing ring", "hex_nut": "the hex nut", "flange_bolt": "the flange bolt",
    "microchip": "the microchip", "cartridge_fuse": "the cartridge fuse", "relay": "the relay",
    "harness_plug": "the harness plug",
    "bottom_bun": "the bottom bun", "burger_patty": "the patty", "cheese_slice": "the cheese slice",
    "tomato_slice": "the tomato slice", "top_bun": "the top bun", "lettuce_leaf": "the lettuce leaf",
    "coffee_mug": "the mug", "coffee_pod": "the gold coffee capsule", "red_capsule": "the red coffee capsule",
    "conical_tube": "the conical tube", "micro_tube": "the micro tube", "micropipette": "the micropipette",
    "centrifuge_lid": "the centrifuge lid", "sponge": "the sponge",
    "red_block": "the red block", "green_block": "the green cylinder", "blue_block": "the blue block",
    "estop": "the red emergency stop button", "cycle_start": "the green start button", "vise": "the vise jaw",
    "test_start": "the green test button", "grill_power": "the grill power button", "brew": "the brew button",
    "centrifuge_start": "the centrifuge start button", "drawer": "the drawer",
}

# Target points: (preposition, noun phrase).
PLACES: dict[str, tuple[str, str]] = {
    "jig_bore": ("in", "the jig bore"), "jig_pin_hole": ("in", "the silver pin hole of the jig"),
    "jig_bolt_hole": ("in", "the blue bolt hole of the jig"), "inbound_tray": ("in", "the inbound tray"),
    "feeder_spindle": ("on", "the feeder spindle"), "feeder_cup": ("in", "the feeder cup"),
    "pallet_red": ("in", "the red compartment of the pallet"),
    "pallet_yellow": ("in", "the yellow compartment of the pallet"),
    "pallet_green": ("in", "the green compartment of the pallet"),
    "pallet_blue": ("in", "the blue compartment of the pallet"), "parts_bin": ("in", "the parts bin"),
    "threaded_stud": ("on", "the threaded stud"), "staging_square": ("on", "the yellow staging square"),
    "chip_socket": ("in", "the chip socket"), "relay_base": ("in", "the relay base"),
    "fuse_clips": ("in", "the fuse clips"), "receptacle": ("in", "the harness receptacle"),
    "parts_tray": ("in", "the parts tray"), "reject_bin": ("in", "the reject bin"),
    "assembly_board": ("on", "the cutting board"), "board_left": ("on", "the cutting board"),
    "board_right": ("on", "the cutting board"), "grill": ("on", "the grill"), "dining_plate": ("on", "the dining plate"),
    "plate_left": ("on", "the dining plate"), "plate_right": ("on", "the dining plate"),
    "caddy_front": ("in", "the bottom-bun slot of the bun caddy"), "caddy_rear": ("in", "the top-bun slot of the bun caddy"),
    "topping_rear": ("in", "the cheese compartment of the topping tray"),
    "topping_front": ("in", "the tomato compartment of the topping tray"), "green_mark": ("onto", "the green mark"),
    "drip_tray": ("on", "the drip tray"), "serving_tray": ("on", "the serving tray"), "tray_left": ("on", "the serving tray"),
    "tray_right": ("on", "the serving tray"), "mug_bin": ("in", "the mug bin"), "pod_caddy": ("in", "the capsule caddy"),
    "capsule_slot": ("into", "the capsule slot"), "saucer": ("on", "the saucer"),
    "rack_a_1": ("in", "the micro-tube hole of the blue rack"), "rack_a_2": ("in", "the middle hole of the blue rack"),
    "rack_a_3": ("in", "the end hole of the blue rack"), "rack_b_conical": ("in", "the orange rack"),
    "rack_b_micro": ("in", "the orange rack"), "centrifuge_slot": ("in", "the centrifuge rotor"),
    "centrifuge_top": ("on", "the centrifuge"), "microplate": ("above", "the microplate"),
    "pipette_pad": ("on", "the blue pipette pad"), "waste_bin": ("in", "the waste bin"),
    "red_bin": ("in", "the red bin"), "green_bin": ("in", "the green bin"), "blue_bin": ("in", "the blue bin"),
    "red_bin_a": ("in", "the red bin"), "red_bin_b": ("in", "the red bin"), "green_bin_a": ("in", "the green bin"),
    "green_bin_b": ("in", "the green bin"), "blue_bin_a": ("in", "the blue bin"), "blue_bin_b": ("in", "the blue bin"),
    "black_square": ("on", "the black square"), "yellow_square": ("on", "the yellow square"),
    "purple_square": ("on", "the purple square"), "drawer_inside": ("in", "the drawer"),
}


def place_phrase(target: str) -> str:
    """'on the grill' for a target point, 'on the patty' for a free body."""
    if target in PLACES:
        prep, noun = PLACES[target]
        return f"{'on' if prep == 'above' else prep} {noun}"
    return f"on {THINGS[target]}"


def _noun(name: str) -> str:
    return PLACES[name][1] if name in PLACES else THINGS[name]


def step_text(goal) -> str:
    if isinstance(goal, On):
        return f"put {THINGS[goal.body]} {place_phrase(goal.target)}"
    if isinstance(goal, Pressed):
        return f"press {THINGS[goal.button]}"
    if isinstance(goal, Closed):
        return f"close {THINGS[goal.slider]}"
    if isinstance(goal, Opened):
        return f"open {THINGS[goal.slider]}"
    if isinstance(goal, Slid):
        return f"slide {THINGS[goal.body]} onto {_noun(goal.target)} without lifting it"
    if isinstance(goal, Lifted):
        return f"lift {THINGS[goal.body]} {goal.height * 100:.0f} cm and hold it there"
    if isinstance(goal, GripperAbove):
        return f"hold the gripper {goal.height * 100:.0f} cm above {_noun(goal.target)}"
    if isinstance(goal, Touching):
        return f"touch the top of {THINGS[goal.body]} with the closed gripper"
    if isinstance(goal, HeldOver):
        return f"hold {THINGS[goal.body]} with its tip {goal.clearance * 100:.0f} cm above {_noun(goal.target)}"
    raise TypeError(f"no wording for {type(goal).__name__}")


def step_by_step(spec: TaskSpec) -> str:
    """The goals as numbered steps, in the order the instruction gives them."""
    steps = [step_text(g) for g in spec.goals]
    if len(steps) == 1:
        return steps[0][0].upper() + steps[0][1:] + "."
    return " ".join(f"{k}. {s[0].upper()}{s[1:]}." for k, s in enumerate(steps, start=1))


def prompt_variants(spec: TaskSpec) -> dict[str, str]:
    return {"canonical": spec.instruction, "step_by_step": step_by_step(spec)}


def missing_names() -> list[str]:
    """Scene entities that tasks refer to but that have no wording here (must be empty)."""
    from afford_pro.bench.tasks import TASKS

    missing = []
    for spec in TASKS:
        for goal in spec.goals:
            try:
                step_text(goal)
            except KeyError as exc:
                missing.append(f"{spec.task_id}: {exc}")
    return missing
