"""Domain 3, Tabletop & Domestic (tasks 61-90): sorting, marked squares, stacking, the drawer, touching and sliding."""
from __future__ import annotations

from afford_pro.bench.spec import Closed, On, Region, Slid, Touching
from afford_pro.bench.tasks._build import FIXED_POINT, task

TT = "tabletop_sorting"
# The sponge waits on the yellow square; tasks that use that square move it to the purple one.
SPONGE_ON_PURPLE = {"sponge": Region((0.30, 0.12), (0.015, 0.018), yaw_deg=10.0)}
SQUARE = "the legacy task's goal was an unmarked table point; it is now a marked square"
SWAP = "the legacy task named the start position of another block, which is not where that block starts in this scene"
RELATIVE = "the legacy task asked to advance the block 10 cm but scored a fixed point; it now slides to a marked square"
BLOCK = {"red": "red_block", "green": "green_block", "blue": "blue_block"}
NOUN = {"red": "red block", "green": "green cylinder", "blue": "blue block"}


def _sort(number: int, color: str, bin_color: str) -> object:
    task_id = f"tabletop_sort_{color}" if color == bin_color else f"tabletop_{color}_to_{bin_color}_bin"
    return task(number, task_id, TT, f"Put the {NOUN[color]} in the {bin_color} bin.",
                On(BLOCK[color], f"{bin_color}_bin"))


TASKS = [
    _sort(61, "red", "red"), _sort(62, "green", "green"), _sort(63, "blue", "blue"),
    _sort(64, "red", "green"), _sort(65, "red", "blue"), _sort(66, "green", "red"),
    _sort(67, "green", "blue"), _sort(68, "blue", "red"), _sort(69, "blue", "green"),
    task(70, "tabletop_red_to_black_square", TT, "Put the red block on the black square.",
         On("red_block", "black_square"), change=SQUARE),
    task(71, "tabletop_green_to_black_square", TT, "Put the green cylinder on the black square.",
         On("green_block", "black_square"), change=SQUARE),
    task(72, "tabletop_blue_to_black_square", TT, "Put the blue block on the black square.",
         On("blue_block", "black_square"), change=SQUARE),
    task(73, "tabletop_stack_green_on_red", TT, "Stack the green cylinder on the red block.",
         On("green_block", "red_block"), change=SWAP),
    task(74, "tabletop_stack_red_on_green", TT, "Stack the red block on the green cylinder.",
         On("red_block", "green_block"), change=SWAP),
    task(75, "tabletop_stack_blue_on_green", TT, "Stack the blue block on the green cylinder.",
         On("blue_block", "green_block"), change=SWAP),
    task(76, "tabletop_stack_red_on_blue", TT, "Stack the red block on the blue block.", On("red_block", "blue_block")),
    task(77, "tabletop_stack_green_on_blue", TT, "Stack the green cylinder on the blue block.",
         On("green_block", "blue_block")),
    task(78, "tabletop_stack_blue_on_red", TT, "Stack the blue block on the red block.", On("blue_block", "red_block")),
    task(79, "tabletop_close_drawer", TT, "Close the drawer.", Closed("drawer"),
         change=f"{FIXED_POINT}; a drawer was added"),
    task(80, "tabletop_red_in_drawer", TT, "Put the red block in the drawer.", On("red_block", "drawer_inside"),
         change=f"{FIXED_POINT}; a drawer was added"),
    task(81, "tabletop_blue_in_drawer", TT, "Put the blue block in the drawer.", On("blue_block", "drawer_inside"),
         change=f"{FIXED_POINT}; a drawer was added"),
    task(82, "tabletop_red_touch", TT, "Touch the top of the red block with the closed gripper.", Touching("red_block"),
         change="the legacy task scored a fixed point above the block's nominal start; the fingertips must now touch the block"),
    task(83, "tabletop_green_touch", TT, "Touch the top of the green cylinder with the closed gripper.",
         Touching("green_block"),
         change="the legacy task scored a fixed point above the block's nominal start; the fingertips must now touch the block"),
    task(84, "tabletop_blue_touch", TT, "Touch the top of the blue block with the closed gripper.", Touching("blue_block"),
         change="the legacy task scored a fixed point above the block's nominal start; the fingertips must now touch the block"),
    task(85, "tabletop_red_to_yellow_square", TT, "Put the red block on the yellow square.",
         On("red_block", "yellow_square"), starts=SPONGE_ON_PURPLE, change=SQUARE),
    task(86, "tabletop_blue_to_purple_square", TT, "Put the blue block on the purple square.",
         On("blue_block", "purple_square"), change=SQUARE),
    task(87, "tabletop_slide_red", TT, "Slide the red block onto the yellow square without lifting it.",
         Slid("red_block", "yellow_square"), starts=SPONGE_ON_PURPLE, change=RELATIVE),
    task(88, "tabletop_slide_blue", TT, "Slide the blue block onto the purple square without lifting it.",
         Slid("blue_block", "purple_square"), change=RELATIVE),
    task(89, "tabletop_wipe_table", TT,
         "Wipe the table: slide the sponge onto the purple square without lifting it.",
         Slid("sponge", "purple_square"), change="the legacy task scored hovering over a point; a sponge and marked squares were added"),
    task(90, "tabletop_green_in_drawer_close", TT, "Put the green cylinder in the drawer and close the drawer.",
         On("green_block", "drawer_inside"), Closed("drawer"), change=f"{FIXED_POINT}; a drawer was added"),
]
