# AFFORD-Bench reality-check hardening: changes, verification, and open issues

Date: 2026-10-08. Branch `claude/intelligent-meitner-e8vp2p`. Measured with MuJoCo 3.2.7 (Python 3.13), rendering disabled, Panda meshes from MuJoCo Menagerie and a builtin checker texture in place of the absolute `D:/` asset paths (object geometry, controller, oracle, and success logic are the repository's).

## Result against the requested targets

| Target | Measured (`docs/reviews/sanity_baselines_2026-10-08.csv`) |
| --- | --- |
| Oracle > 95% (nominal) | 1014/1050 episodes, 96.6% (episode-level Wilson 95% CI 95.3 to 97.5%), 5 reset seeds per task; 192/210 tasks solved in every seed, 209/210 in at least one |
| Null policy 0.0% | 0/210 tasks (1 episode per task; upper 95% bound 1.8%) |
| Random policy 0.0% | 0/1050 episodes (5 per task, H = 400; upper 95% bound 0.4%) |
| No task solved at reset | 0/210 (3 reset seeds per task) |
| Oracle under L4 (not a target; reported) | 509/1050 episodes, 48.5% (45.5 to 51.5%) |

Per domain (oracle nominal / oracle L4, % of 150 episodes): D1 100.0 / 42.7, D2 92.7 / 52.7, D3 100.0 / 44.7, D4 100.0 / 88.0, D5 94.7 / 38.7, D6 95.3 / 50.7, D7 93.3 / 22.0.

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

## Remaining oracle failures (36 of 1050 nominal episodes)

29 are mug, bun, and capsule episodes in D2, D5, and D7. In the traced episodes the grasp was lost during transport and the body landed on its side; the oracle grasps top-down only and cannot re-orient a toppled body. Task 168 fails in all 5 seeds (the released pin settles 3.2 mm from its goal against a 2.5 mm tolerance) and task 155 in 2 of 5.

## Reproduction

```
python scripts/sanity_baselines.py --asset-dir <portable assets> --shard i/4 --oracle-episodes 5 --episodes 5 --out part_i.csv
python scripts/l4_slip_probe.py --asset-dir <portable assets> --tasks tabletop_sort_red,kitchen_mug_return_bin,industrial_shaft_to_pallet --out slip.csv
pytest tests
```
