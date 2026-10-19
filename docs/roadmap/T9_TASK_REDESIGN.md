# T9: every task valid end to end (AFFORD-Bench tasks)

Goal (authors, 2026-10-18): all 210 tasks accurate in the sense of LIBERO / LIBERO-PRO / LIBERO-Plus / LIBERO-X: the
instruction names simulated objects, the goal is a conjunction of predicates over them, and the success check covers
the whole instruction. Acceptance gate: ROADMAP.md, T9. The per-task definitions are in `T9_TASK_TABLE.md`
(generated from the code by `scripts/export_task_table.py`).

## What was built
- **Task specification** (`afford_pro/bench/spec.py`). A task is an instruction, start overrides, and goals:
  `On(body, target)` (released, supported, at rest, upright, at a target point, inside a region, or on another
  body), `Pressed(button)`, `Closed(slider)`, `Slid(body, target)` (moved along its support, never lifted more than
  1 cm), `Lifted(body, h)`, `HeldOver(body, target, clearance)`, `GripperAbove(target, h)`, `Touching(body)`.
  Success = every goal holds at the same step; a goal counts only once the goals ordered before it have been reached;
  waypoints (`once`) and events (a touch after an insertion) must have held once. 210 specifications in
  `afford_pro/bench/tasks/`, one module per domain; static checks in `afford_pro/bench/tasks/__init__.py`.
- **Scenes** (`scripts/build_bench_scenes.py` + `scripts/bench_scene_parts.py` generate `afford_pro/assets/*_bench.xml`
  from the source scenes; `tests/test_bench_tasks.py` fails if the XMLs are stale). Added:
  - industrial assembly: bearing ring, hex nut and threaded stud, flange bolt and a blue-rimmed bolt hole, a sliding
    vise jaw, a 2x2 colored pallet, a parts bin, a parts mat, a yellow staging square, a latching e-stop and a green
    start button;
  - new electronics bench: microchip and socket, cartridge fuse and clips, relay and base, harness plug and
    receptacle, parts tray, reject bin, test button;
  - burger kitchen: lettuce leaf, sponge and wiping lane, grill power button; low rims on the bun caddy and the
    topping tray;
  - coffee kitchen: second (red) capsule, capsule slot on the machine, saucer, brew button on top, sponge;
  - lab: domed centrifuge lid, centrifuge start button, waste bin, blue pipette pad, sponge, shorter micro-tube holes;
  - tabletop: drawer in a cabinet, black, yellow and purple squares, sponge.
  Buttons are slide joints with dry friction and gravity compensation (they stay where pushed: the state is in the
  physics state); sliders (drawer, vise) likewise.
- **Environment** (`afford_pro/bench/env.py`, `GoalFrankaEnv`). At start-up every goal is made physically real: each
  goal body is dropped into its target (or slid down the axis of the body it goes on) and must come to rest there,
  else the task fails to build. Each task's scene holds the bodies it names plus two distractors from the scene's
  pool (LIBERO's convention). Starts are sampled per body (regions with yaw, slots, or pre-placed assemblies) with
  rejection: no overlap, a collision-free grasp from the oracle's own candidate set, no fixture overhead, no goal
  already satisfied. Region goals (bins, trays, plates, squares) accept any pose with the footprint inside; fits
  (bores, holes, sockets, studs, racks) keep the scene precision (2 mm industrial and lab, 8 mm kitchens and
  tabletop). The IK adds a null-space pull to the ready posture (AUDIT A-33).
- **Oracle** (`afford_pro/bench/oracle.py`): the grasp-carry-place routine of `ParametricOracle` applied goal by goal, plus primitives for
  buttons, sliders, wiping, lifting, holding over a target and touching; parts with a socket orientation are turned
  to it during the carry; a part that stops within 1.5 mm of its seat is released there.
- **Language** (`afford_pro/bench/language.py`): the canonical instruction is hand-written; a step-by-step variant is
  derived from the goals, so it states exactly what is checked. Datasets store these, not the legacy prompts.
- **Pipeline**: `afford_pro.make(task)` builds the benchmark task (the legacy set: `task_set="legacy"`, deprecated);
  files record their task set; a benchmark file's replay must reproduce every generalized coordinate exactly.
  New `objects` modality (proprioception + poses of the moved bodies, 6 slots).

## Decisions (with reasons)
- Unsimulable steps (liquids, cutting, cooking, deformables, flipping a patty with a top-down gripper) are replaced
  by the closest manipulation the scene can simulate; every replacement is listed with its reason.
- Start locations are named in an instruction only when the start is a receptacle the spatial suite keeps (the
  spatial suite moves region starts 7-12 cm).
- Pre-placed assemblies (shaft in the jig, burger layers, tube in the centrifuge) are fixed starts by definition; 31
  goal bodies start this way and 22 start in one of two slots, neither of which is their own first goal (a static
  check). The 225 region starts have s.d. >= 0.99 cm over 20 resets.
- "Into the bin" goals (the six that allow any tilt) accept the body in any pose: lowest point on the bin floor,
  origin inside the bin.
- Contact model (benchmark scenes only): objects touch each other with condim 3 (the finger pads keep condim 4), and no thin
  collision layer lies inside another where things are set down (decorative layers are visual only). Thin light parts
  released under L4/L5 still sink through their support in 0.5-1.6% of oracle episodes (A-42).
- The mug stays the source scene's solid cylinder: a hollow thin-walled mug let a finger slip inside and lift it by the rim, so the
  two "drop a capsule into the mug" tasks were redefined.
- The bearing ring can rest up to 5.5 mm off the shaft axis (its bore rides on the shaft core), so "over the shaft" is
  checked to 6 mm, which still requires the ring to encircle the shaft.

## Status
- [x] T9.1 goal schema   - [x] T9.2 environment   - [x] T9.3 oracle   - [x] T9.4 assets
- [x] T9.5 task redefinitions (`T9_TASK_TABLE.md`)   - [x] T9.6 validation of all 210 (E-016, E-017)
- [x] T9.7 language (canonical + derived step-by-step)   - [ ] T9.8 datasets (deferred by the authors)
- [ ] T9.9 paper, page, card
