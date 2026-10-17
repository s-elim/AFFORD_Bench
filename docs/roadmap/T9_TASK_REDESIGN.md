# T9 design: every task valid end to end

Goal (authors, 2026-10-18): all 210 tasks accurate in the sense of LIBERO / LIBERO-PRO / LIBERO-X / LIBERO-Plus: the
instruction names simulated objects, the goal is a conjunction of predicates over them, and the success check covers
the whole instruction. Acceptance gate: ROADMAP.md, T9.

## Current scene inventory (2026-10-18)
| Scene | Movable bodies | Fixtures | Mechanisms |
| --- | --- | --- | --- |
| industrial_assembly | machined_shaft, spur_gear, dowel_pin (+2 distractors) | workstation, andon_tower, assembly_jig, inbound_tray, gear_feeder, pin_feeder, outbound_pallet | none |
| kitchen_burger | bottom_bun, burger_patty, cheese_slice, tomato_slice, top_bun (+2) | counter, assembly_board, grill_station, bun_caddy, topping_tray, dining_plate | none |
| kitchen_coffee | coffee_mug, coffee_pod (+2) | table, counter_mat, coffee_machine, mug_bin, pod_caddy, serving_tray | none |
| lab_science | conical_tube, micro_tube, micropipette (+2) | bench, microcentrifuge, tube_rack_a, tube_rack_b, microplate_96 | none |
| tabletop_sorting | red_block, green_block, blue_block (+2) | table, red_bin, green_bin, blue_bin | none |

16 movable bodies and no articulated mechanism support 210 tasks today; that is why 144 instructions name things that
do not exist or describe actions that are not simulated.

## Approach
1. **Goal schema (T9.1-T9.3).** `TaskDef.goals`: ordered list of predicates, each *placed(body, on/in support,
   pose, tol)*, *held(body, pose)*, *pressed(joint, depth)*, *opened/closed(joint, range)*, *touched(body)*, or
   *hovered(point relative to a body, steps)*. Success = all predicates true at the same step. The oracle executes
   the list in order with its existing grasp-carry-release routine, and push/press primitives for mechanisms.
2. **Add the objects and mechanisms the instructions need (T9.4)**, built from MuJoCo primitives with real
   clearances, and validated like the existing bodies (graspable, stable at rest, oracle >= 95%):
   - industrial: bearing ring, hex nut, flange bolt, PCB chip + socket, fuse + holder, relay + base, cable plug +
     port; sprung e-stop button; toggle clamp (hinge).
   - kitchen_burger: grill control knob or button; spatula only if a flip can be executed reliably, otherwise the
     flip step is redefined.
   - kitchen_coffee: sprung brew button; second cup or saucer for service chains.
   - lab_science: centrifuge START button and hinged lid, pipette plunger (slide), tube caps.
   - tabletop: drawer (slide) for the "drawer" instructions; touch predicates on blocks.
3. **Physically impossible steps** (liquids, cracking, cutting, freezing, heating) cannot be simulated with rigid
   bodies; those instructions are redefined to the closest physically simulated manipulation in the same scene,
   and the change is recorded per task (old instruction, new instruction, reason).
4. **Reach tasks** become real mechanism or contact tasks where a mechanism is added (press, touch), or honest
   positioning tasks ("hold the gripper 12 cm above the red bin") whose predicate is exactly that.
5. **Chains (D5-D7)** become 2-3 predicate conjunctions over bodies present in their scene (e.g. task 121: shaft in
   jig bore AND gear on shaft AND pin in locating hole), so the episode cannot end before the instruction is done.

## Status
- [ ] T9.1 goal schema   - [ ] T9.2 environment   - [ ] T9.3 oracle   - [ ] T9.4 assets
- [ ] T9.5 task redefinitions (per-task table: old, new, predicates, reason)   - [ ] T9.6 validation of all 210
- [ ] T9.7 prompts   - [ ] T9.8 datasets_v2   - [ ] T9.9 paper, page, card
