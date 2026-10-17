"""Task tiers: which tasks measure what their instruction says.

The tier of a task is computed from its definition and the per-task grounding flags in
`configs/prompts_augmented.yaml` (checked against the task definitions by tests/test_prompts_augmented.py), so it
cannot drift from the code:

- core:                  a pick, place, or insertion task whose instruction names the simulated body and whose
                         success predicate covers the whole instruction.
- reach:                 the TCP must hover within tolerance of a fixed world point for 5 control steps
                         (task_type "press_button"); no button, wipe, or seal is simulated and the point does not
                         move with the scene. Report separately from manipulation success.
- instruction_mismatch:  the instruction names a body that is not simulated, or the episode ends before the
                         instruction is complete (single-body success in a multi-step instruction).

Use `tasks_in_tier("core")` for any claim about manipulation with language grounding.
"""
from __future__ import annotations

from functools import lru_cache

from afford_pro.task_definitions import ALL_210_TASKS, load_all_augmented_prompts

TIERS = ("core", "reach", "instruction_mismatch")

_HOVER_VS_REST = "the instruction asks to hold the pipette tip above the target; success requires releasing the pipette to stand on it"
_SWAP_GOAL = ("the goal is the named block's default position, but that block is placed elsewhere in this scene, "
              "so the goal is an unmarked table point")
_RELATIVE_PUSH = "'advance it 10 cm forward' is relative to a start scattered by up to 7 cm, but the goal is a fixed world point"
_NO_STACK = "the instruction says 'onto the burger stack'; no stack exists in this scene, the goal is a point on the bare board"

# Mismatches found by the runtime task audit (docs/roadmap/evidence/task_audit_2026-10-17.json) that the prompt-file
# flags do not capture. Each entry overrides "core".
AUDITED_MISMATCHES: dict[str, str] = {
    "kitchen_tomato_place": _NO_STACK,
    "kitchen_top_bun_crown": _NO_STACK,
    "kitchen_patty_flip": "the instruction asks to sear the reverse side (a flip); the goal is the start pose, upright",
    "tabletop_swap_red_blue": _SWAP_GOAL,
    "tabletop_swap_green_red": _SWAP_GOAL,
    "tabletop_swap_blue_green": "the goal calibrates to a hold goal (the block drifts 23 mm when released there), so success "
                                "requires holding the block, not placing it",
    "tabletop_push_red_forward": _RELATIVE_PUSH,
    "tabletop_push_blue_forward": _RELATIVE_PUSH,
    "lab_pipette_to_microplate": _HOVER_VS_REST,
    "lab_pipette_to_tube_mouth": _HOVER_VS_REST,
    "lab_pipette_to_well_corner": _HOVER_VS_REST,
    "lab_pipette_to_centrifuge_tube": _HOVER_VS_REST,
}


# Core tasks to report separately (runtime task audit, docs/roadmap/evidence/task_audit_2026-10-17.json):
# the goal is an unmarked point on a table, bench, board, or workstation (recall of a world coordinate) ...
UNMARKED_GOAL_CORE: tuple[str, ...] = (
    "industrial_clear_feeder_spindle", "industrial_clear_inbound_tray", "kitchen_bottom_bun_to_board",
    "kitchen_top_bun_to_board", "kitchen_tomato_to_board", "kitchen_cheese_to_board", "tabletop_red_center_table",
    "tabletop_green_center_table", "tabletop_blue_center_table", "tabletop_clear_left_quadrant",
    "tabletop_clear_right_quadrant", "lab_pick_micropipette", "lab_pipette_dock_stand", "lab_conical_tube_to_plate",
    "lab_micro_tube_to_plate", "lab_tube_rack_b_to_center", "lab_micro_tube_to_center",
)
# ... or the start of the evaluated body varies by less than 1 cm (std over 10 audit resets), so a fixed trajectory
# can succeed.
NEAR_FIXED_START_CORE: tuple[str, ...] = (
    "industrial_gear_mesh", "industrial_gear_to_pallet", "industrial_gear_to_tray", "industrial_gear_flip",
    "industrial_tray_kitting_slot2", "industrial_clear_feeder_spindle", "kitchen_cheese_place",
    "kitchen_bottom_bun_to_board", "kitchen_tomato_to_plate", "kitchen_cheese_to_plate", "kitchen_top_bun_to_board",
    "kitchen_tomato_to_board", "kitchen_cheese_to_board",
)
CORE_SUBSETS: dict[str, tuple[str, ...]] = {"unmarked_goal": UNMARKED_GOAL_CORE, "near_fixed_start": NEAR_FIXED_START_CORE}


@lru_cache(maxsize=1)
def _tier_map() -> dict[str, str]:
    prompts = load_all_augmented_prompts()
    if not prompts:
        raise RuntimeError("configs/prompts_augmented.yaml is missing or unreadable; task tiers cannot be computed")
    tiers: dict[str, str] = {}
    for task in ALL_210_TASKS:
        if task.task_type == "press_button":
            tiers[task.task_id] = "reach"
            continue
        if task.task_id not in prompts:
            raise RuntimeError(f"{task.task_id} has no entry in configs/prompts_augmented.yaml; its tier is undefined")
        grounding = prompts[task.task_id].get("sim_grounding", {})
        grounded = bool(grounding.get("referent_in_scene")) and bool(grounding.get("instruction_covered_by_success"))
        grounded = grounded and task.task_id not in AUDITED_MISMATCHES
        tiers[task.task_id] = "core" if grounded else "instruction_mismatch"
    return tiers


def mismatch_reason(task_id: str) -> str | None:
    """Why a task is not core (None for core and reach tasks)."""
    if task_tier(task_id) != "instruction_mismatch":
        return None
    if task_id in AUDITED_MISMATCHES:
        return AUDITED_MISMATCHES[task_id]
    grounding = load_all_augmented_prompts()[task_id]["sim_grounding"]
    if not grounding.get("referent_in_scene"):
        return "the instruction names an object that is not the simulated body"
    return "the episode ends when one body reaches its goal, before the instruction is complete"


def task_tier(task_id: str) -> str:
    try:
        return _tier_map()[task_id]
    except KeyError:
        raise KeyError(f"{task_id} is not one of the 210 core-registry tasks") from None


def tasks_in_tier(tier: str) -> list[str]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return [t.task_id for t in ALL_210_TASKS if _tier_map()[t.task_id] == tier]


def tier_counts_by_domain() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for task in ALL_210_TASKS:
        counts = out.setdefault(task.domain, {tier: 0 for tier in TIERS})
        counts[_tier_map()[task.task_id]] += 1
    return out
