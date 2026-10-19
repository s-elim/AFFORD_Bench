"""Domain 2, Culinary & Kitchen Operations (tasks 31-60): single placements, stacking, buttons and wiping in the
burger and coffee kitchens."""
from __future__ import annotations

from afford_pro.bench.spec import On, Placed, Pressed, Slid, Touching
from afford_pro.bench.tasks._build import FIXED_POINT, task

BURGER, COFFEE = "kitchen_burger", "kitchen_coffee"
BUN_ON_BOARD = {"bottom_bun": Placed("assembly_board")}
BASE = {**BUN_ON_BOARD, "burger_patty": Placed("bottom_bun")}
BASE_CHEESE = {**BASE, "cheese_slice": Placed("burger_patty")}
BASE_TOMATO = {**BASE_CHEESE, "tomato_slice": Placed("cheese_slice")}
FULL_BURGER = {**BASE_TOMATO, "top_bun": Placed("tomato_slice")}
NO_STACK = "the legacy task's goal was a point over a bare board; the layers below now start stacked on the board"

TASKS = [
    task(31, "kitchen_patty_on_bun", BURGER, "Put the patty on the bottom bun on the cutting board.",
         On("burger_patty", "bottom_bun"), starts=BUN_ON_BOARD, change=NO_STACK),
    task(32, "kitchen_cheese_on_patty", BURGER, "Put the cheese slice on the patty.",
         On("cheese_slice", "burger_patty"), starts=BASE, change=NO_STACK),
    task(33, "kitchen_tomato_on_cheese", BURGER, "Put the tomato slice on the cheese slice.",
         On("tomato_slice", "cheese_slice"), starts=BASE_CHEESE, change=NO_STACK),
    task(34, "kitchen_top_bun_crown", BURGER, "Crown the burger: put the top bun on the tomato slice.",
         On("top_bun", "tomato_slice"), starts=BASE_TOMATO, change=NO_STACK),
    task(35, "kitchen_bottom_bun_to_board", BURGER, "Put the bottom bun on the cutting board.",
         On("bottom_bun", "assembly_board")),
    task(36, "kitchen_mug_to_drip_tray", COFFEE, "Put the mug on the drip tray of the coffee machine.",
         On("coffee_mug", "drip_tray")),
    task(37, "kitchen_brew_press", COFFEE, "Press the brew button on top of the coffee machine.",
         Pressed("brew"), change="the legacy task scored hovering near a fixed point; a latching brew button was added"),
    task(38, "kitchen_capsule_to_slot", COFFEE, "Put the gold coffee capsule into the capsule slot on the machine.",
         On("coffee_pod", "capsule_slot"), change="the legacy task's hopper was a point on the machine top; a capsule slot was added"),
    task(39, "kitchen_mug_to_serving_tray", COFFEE, "Move the mug from the drip tray to the serving tray.",
         On("coffee_mug", "serving_tray"), starts={"coffee_mug": Placed("drip_tray")}),
    task(40, "kitchen_patty_to_plate", BURGER, "Put the patty on the dining plate.", On("burger_patty", "dining_plate")),
    task(41, "kitchen_tomato_to_plate", BURGER, "Put the tomato slice on the dining plate.",
         On("tomato_slice", "dining_plate")),
    task(42, "kitchen_cheese_to_plate", BURGER, "Put the cheese slice on the dining plate.",
         On("cheese_slice", "dining_plate")),
    task(43, "kitchen_capsule_to_tray", COFFEE, "Put the gold coffee capsule on the serving tray.",
         On("coffee_pod", "serving_tray")),
    task(44, "kitchen_mug_return_bin", COFFEE, "Put the mug from the serving tray back in the mug bin.",
         On("coffee_mug", "mug_bin"), starts={"coffee_mug": Placed("serving_tray")}),
    task(45, "kitchen_capsule_return_caddy", COFFEE,
         "Put the gold coffee capsule from the serving tray back in the capsule caddy.",
         On("coffee_pod", "pod_caddy"), starts={"coffee_pod": Placed("serving_tray")}),
    task(46, "kitchen_grill_power", BURGER, "Turn on the grill with its power button.",
         Pressed("grill_power"),
         change="the legacy task asked to flip the patty but scored its start pose; a flip needs a regrasp a top-down gripper "
                "cannot make, so the task actuates the grill instead"),
    task(47, "kitchen_top_bun_to_board", BURGER, "Put the top bun on the cutting board.", On("top_bun", "assembly_board")),
    task(48, "kitchen_tomato_to_board", BURGER, "Put the tomato slice on the cutting board.",
         On("tomato_slice", "assembly_board")),
    task(49, "kitchen_cheese_to_board", BURGER, "Put the cheese slice on the cutting board.",
         On("cheese_slice", "assembly_board")),
    task(50, "kitchen_touch_top_bun", BURGER, "Gently touch the top bun of the assembled burger.",
         Touching("top_bun"), starts=FULL_BURGER,
         change="the legacy task scored a fixed point over a bare board; the burger now stands there and the fingertips must touch "
                "its top bun without moving it"),
    task(51, "kitchen_mug_to_saucer", COFFEE, "Put the mug on the saucer.", On("coffee_mug", "saucer"),
         change=FIXED_POINT),
    task(52, "kitchen_lettuce_to_board", BURGER, "Put the lettuce leaf on the cutting board.",
         On("lettuce_leaf", "assembly_board"), change=FIXED_POINT),
    task(53, "kitchen_bottom_bun_to_caddy", BURGER, "Put the bottom bun back in its slot in the bun caddy.",
         On("bottom_bun", "caddy_front"), starts={"bottom_bun": Placed("assembly_board")}),
    task(54, "kitchen_top_bun_to_caddy", BURGER, "Put the top bun back in its slot in the bun caddy.",
         On("top_bun", "caddy_rear"), starts={"top_bun": Placed("assembly_board")}),
    task(55, "kitchen_cheese_to_topping_tray", BURGER, "Put the cheese slice back in its compartment of the topping tray.",
         On("cheese_slice", "topping_rear"), starts={"cheese_slice": Placed("assembly_board")}),
    task(56, "kitchen_tomato_to_topping_tray", BURGER,
         "Put the tomato slice back in its compartment of the topping tray.",
         On("tomato_slice", "topping_front"), starts={"tomato_slice": Placed("assembly_board")}),
    task(57, "kitchen_patty_to_grill", BURGER, "Put the patty back on the grill.",
         On("burger_patty", "grill"), starts={"burger_patty": Placed("assembly_board")}),
    task(58, "kitchen_wipe_counter", BURGER,
         "Wipe the counter with the sponge: slide it onto the green mark without lifting it.",
         Slid("sponge", "green_mark"), change="the legacy task scored hovering over a point; a sponge and a marked lane were added"),
    task(59, "kitchen_wipe_coffee_counter", COFFEE,
         "Slide the sponge along the counter onto the green mark without lifting it.",
         Slid("sponge", "green_mark"), change="the legacy task scored hovering over the drip tray; a sponge and a marked lane were added"),
    task(60, "kitchen_bottom_bun_to_plate", BURGER, "Put the bottom bun on the dining plate.",
         On("bottom_bun", "dining_plate"), change=FIXED_POINT),
]
