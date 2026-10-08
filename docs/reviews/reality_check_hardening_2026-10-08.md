# AFFORD-Bench reality-check hardening: changes, verification, and open issues

Date: 2026-10-08. Branch `claude/intelligent-meitner-e8vp2p`. Measured with MuJoCo 3.2.7 (Python 3.13), rendering disabled, Panda meshes from MuJoCo Menagerie and a builtin checker texture in place of the absolute `D:/` asset paths (object geometry, controller, oracle, and success logic are the repository's).

## Result against the requested targets

| Target | Measured (`docs/reviews/sanity_baselines_2026-10-08.csv`) |
| --- | --- |
| Oracle > 95% (nominal; follow-up target >= 99%) | 1048/1050 episodes, 99.8% (episode-level Wilson 95% CI 99.3 to 99.9%), 5 reset seeds per task; 208/210 tasks solved in every seed, 210/210 in at least one |
| Null policy 0.0% | 0/210 tasks (1 episode per task; upper 95% bound 1.8%) |
| Random policy 0.0% | 0/1050 episodes (5 per task, H = 400; upper 95% bound 0.4%) |
| No task solved at reset | 0/210 (3 reset seeds per task) |
| Oracle under L4 (not a target; reported) | 497/1050 episodes, 47.3% (44.3 to 50.4%) |

Per domain (oracle nominal / oracle L4, % of 150 episodes): D1 100.0 / 37.3, D2 100.0 / 56.7, D3 100.0 / 41.3, D4 100.0 / 84.0, D5 100.0 / 38.0, D6 99.3 / 51.3, D7 99.3 / 22.7. The first hardening run measured 96.6% nominal and 48.5% under L4; the follow-up below raised the nominal rate.

The Wilson intervals treat episodes as independent; episodes of one task are correlated, so the intervals are too narrow. A single simulator build was used; no cross-version or cross-machine check was run.

## What changed, by requested item

1. **Tolerances, no snapping.** Tiers in `task_definitions.PRECISION_TIERS`: D1 and D4 2.0 mm and 5 deg, D6 2.5 mm and 5 deg, D2, D3, D7 8.0 mm and 15 deg; D5 takes the tier of its scene (2 mm in the industrial and lab scenes, 8 mm in the kitchen and tabletop scenes). All attachment, release snapping, and goal teleport are gone. Success (`ParametricFrankaEnv.check_success`) requires the body to be released (no finger contact), supported by a non-robot body, at rest (|v| < 0.02 m/s, |w| < 0.2 rad/s), within the xy, z, and tilt tolerance of the goal, and lifted at least `min_lift` (2 cm for every manipulation task, 10 to 15 cm for four lift-and-replace tasks). The goal is the settled resting pose of the body at the target, found at construction by lowering the body onto its support and simulating to rest; targets without a stable resting pose are hold goals (two-finger grasp at the goal). Press tasks need the TCP within the tier tolerance for 5 consecutive steps.
2. **Scatter.** L1: start offset U[-7, 7] cm per axis, yaw U[-45, 45] deg, spawn roll/pitch U[-10, 10] deg resolved by settling, TCP jitter U[-1.5, 1.5] cm. Starts are rejection-sampled for physical validity (see deviations). The oracle reads the body pose every step.
3. **Distractors.** Two `distractor_*` free bodies per scene (similar color or shape), placed in D1, D3, D7 tasks at least 9 cm from start and goal.
4. **L4.** `p4_physical`: sliding friction U[0.12, 0.25] on the body and both finger pads (MuJoCo combines a pair by the maximum, so both are set) and mass and inertia x U[1.8, 2.5]. Nominal parameters are restored each episode.
5. **Paper.** Abstract, contributions, Sec. III action and success set, Table I tolerances, Sec. IV (all levels, physics), Sec. V (closed-loop oracle), H2 range, Table II, Sec. VII, hardware gaps, and the Stage 2 charter are updated; `main.pdf` recompiled (body 5 pages, references 2, no overfull boxes).

Physics and control changes that the above required: gravity-compensated joint position servos with velocity feedforward (20 Hz, 25 substeps of 2 ms), damped-least-squares IK with commanded yaw, a force-limited tendon gripper (about 35 N per finger) whose opening is slew-limited to 0.05 m/s per finger, stiff finger-pad contacts, native CCD with multi-point contact, torsional and rolling friction normalized, hollow receptacles built from box rings with 1.5 mm radial clearance (the ring `euler` angles were previously written in degrees into radian-mode files and are fixed), and a closed-loop oracle with collision-checked grasp planning, grasp verification and retries, in-hand offset correction, insertion jam recovery, and verified release.

## L4 slip evidence (`scripts/l4_slip_probe.py`, `docs/reviews/l4_slip_probe_2026-10-08.csv`)

Oracle grasp shifted 0, 10, or 20 mm along the hand's lateral axis, lift 10 cm, carry 20 cm at the speed cap, 3 seeds per cell.

| Body | Nominal, off-center (10 or 20 mm) | L4, off-center | L4, centered |
| --- | --- | --- | --- |
| Cube (`tabletop_sort_red`) | held 6/6, turned 2 to 5 deg | held 1/6 | held 3/3, slid 3 to 9 mm |
| Mug (`kitchen_mug_return_bin`) | held 6/6, turned 11 to 26 deg | held 3/6 (10 mm: 0/3, 20 mm: 3/3) | held 3/3, slid 6 to 13 mm |
| Shaft (`industrial_shaft_to_pallet`) | held 0/6 (pads miss the 24 mm cylinder) | held 0/6 | held 0/3 |

So off-center grasps fail far more often under L4 (8/12 dropped for cube and mug against 0/12 nominal), as requested. The mug result is not monotonic in the offset, and the shaft is dropped under L4 even when grasped centrally, so some L4 episodes may be unsolvable for any top-down parallel-jaw policy. The oracle's 48.5% under L4 is therefore not a solvability bound.

## Deviations from the request, and decisions Selim should review

- **Rejection-sampled scatter.** A start is accepted only if the body rests flat on one support (tilt < 3 deg, support spread <= 2 mm), has no fixture overhead, admits a collision-free top-down grasp at one of four yaws, and is away from the goal. Otherwise the scatter is shrunk to 1/2, 1/4, 1/10, then the nominal pose is used (9 of 630 measured resets; task 12, a gear seated on its shaft, always). Bodies that start in compartments therefore vary less than +-7 cm, and the spawn tilt only perturbs the settling (no body starts tilted). Uniform +-7 cm without these checks produced bodies bridging rims, leaning on walls, or wedged where no gripper fits.
- **"Angular tolerance" is tilt from upright.** Yaw at the goal is not checked (cubes accept any face up).
- **L2 redefined** as an annulus (radius 7 to 12 cm, |yaw| 45 to 90 deg) so that it lies outside the enlarged L1 support.
- **Scene and TaskDef edits** made to keep tasks solvable with a Franka hand: coffee drip tray moved forward of the brew head and serving tray moved to (0.35, -0.32); burger caddy and tray rims lowered to 8 mm; cheese slice resized to 56 x 56 x 10 mm and the top bun given a flat base (66 mm); a micro-tube hole added to rack B and rack A hole 1 resized; goals or starts moved for 47 tasks (6, 9, 10, 16, 28, 29, 36, 39, 43, 44, 45, 51, 59, 85, 86, 87, 95, 127, 135, 153, 155, 158, 165, 167, 168, 170, 173, 177, 180, 181, 182, 183, 184, 185, 188, 189, 190, 193, 198, 199, 200, 201, 202, 203, 204, 207, 210). Several moves change the scene semantics of the instruction (for example a start "from the feeder" now scatters off the feeder).
- **Gripper slew limit** of 0.05 m/s per finger was added; without it the fingers closed 37 mm in one 50 ms step and launched thin bodies through their support. The Franka Hand rating used for this value is [unverified].
- **Results depend on the substituted Menagerie meshes.** Finger and hand collision geometry decide many of the grasp outcomes; rerun on the author's assets before quoting numbers.

## Follow-up: oracle patch (36 -> 2 nominal failures)

- **Transport.** The carry legs use 0.012 m per step (0.24 m/s) and 0.8 m/s^2 instead of 0.5 m/s and 1.6 m/s^2, and the carry height clears every surface under the hand's footprint on the straight path to the goal by 25 mm. In the traced capsule failure the cause was not slip: the hand struck the coffee brew head on the way, which the path clearance removes.
- **Placement.** The oracle hovers above the goal until the carried body stops swinging and is within 0.5 mm, re-centers laterally (at most 15 steps) during the last 15 mm of the descent, and holds the grasp until the body is at rest on its support before opening (compliant settling).
- **Grasp yaw.** The mug may be grasped at 45 deg steps around its body, not only aligned with its handle, which frees grasps next to the brew head.
- **Scene.** The top bun started straddling its caddy wall and leaned 8.9 deg whenever the start fell back to the nominal pose; its nominal start and the two caddy targets moved to the compartment center (tasks 34, 47, 54, 125, 144, 147, 186).

Remaining nominal failures: one episode each of task 168 (the released pin settles 3.6 mm from its goal against a 2.5 mm tolerance) and task 203 (not traced). Under L4 the oracle succeeds in 47.3% of episodes; its grip and speed do not adapt to friction or mass.

The robot-scene contact force (net force from the scene, excluding the manipulated body, on the robot) is now reported in every step's `info["robot_scene_force"]`. One oracle episode per task (seed 20000) peaked at 149 N in the worst task (median 0 N), which sets the paper's F_max = 150 N. The same probe gives a mean successful episode length of 108 steps, so 50 demonstrations per task is about 1.1e6 timesteps; the released dataset has not been regenerated with this controller.

## Reproduction

```
python scripts/sanity_baselines.py --asset-dir <portable assets> --shard i/4 --oracle-episodes 5 --episodes 5 --out part_i.csv
python scripts/l4_slip_probe.py --asset-dir <portable assets> --tasks tabletop_sort_red,kitchen_mug_return_bin,industrial_shaft_to_pallet --out slip.csv
python scripts/convert_to_lerobot.py --hdf5 <dataset>.hdf5 --out lerobot/
python scripts/convert_to_rlds.py --hdf5 <dataset>.hdf5 --out rlds/ --format npz   # --format tfds needs tensorflow_datasets (not run here)
pytest tests
```
