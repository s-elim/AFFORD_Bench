# Claim-evidence ledger

Every quantitative or scientific claim in the paper (`AFFORD-Bench_RSS'27/paper-template-latex/main.tex`, kept local
and untracked since 2026-10-19), as rewritten on 2026-10-19 for the benchmark tasks, with its artifact. Status:
SUPPORTED, PARTIALLY SUPPORTED, UNVERIFIED, CONTRADICTED, NOT APPLICABLE (design specification or plan, not a result).
Evidence paths are relative to `docs/roadmap/evidence/`; E-xxx refer to `EXPERIMENT_LOG.md`. All runs: MuJoCo 3.5.0,
code at the commit in each file's provenance.

## Task set

| ID | Claim (section) | Artifact | Reproduce | Status |
| --- | --- | --- | --- | --- |
| C-01 | 210 tasks in 7 domains of 30 over six workcells (abstract, Sec. III-D) | `afford_pro.list_tasks()`, `afford_pro/bench/tasks/` | `pytest tests/test_bench_tasks.py::test_registry_has_210_consistent_tasks` | SUPPORTED |
| C-02 | 343 goals, 1-6 per task; 27 objects, 6 latching buttons, a drawer, a vise jaw; 8 predicate types (abstract, Sec. I, Table I) | `bench_tasks.json`, `afford_pro/bench/scenes.py` | `python scripts/export_task_table.py` | SUPPORTED |
| C-03 | Every instruction names only simulated objects and mechanisms (abstract, Sec. I) | `T9_TASK_TABLE.md` (instruction, goals, starts per task); static checks of every name in goals and starts | `pytest tests/test_bench_tasks.py` (static problems, goal wording) | SUPPORTED (names in goals and starts are checked automatically; the instruction text was audited task by task in T9) |
| C-04 | Success is the conjunction of the goals, reached in order; waypoints need only be reached (Sec. III-B) | `GoalFrankaEnv.check_success`, `_after_step` | `pytest tests/test_bench_tasks.py -k "order or latched"` | SUPPORTED |
| C-05 | Placement tolerances 2 mm/5 deg (precision) and 8 mm/15 deg; 6 mm for two loose fits; any pose for bin goals (abstract, Sec. III-B, Table I) | `afford_pro/bench/scenes.py` (precision), task specs (bearing ring, lid), `GoalFrankaEnv.on_error` | `pytest tests/test_bench_tasks.py -k "region or bin"` | SUPPORTED |
| C-06 | Every goal pose is found by letting the body come to rest, so every goal is attainable (Sec. I, III-B) | `GoalFrankaEnv._calibrate`; 0 build errors in E-016 | E-016 | SUPPORTED |
| C-07 | D1-D4 have one goal per task (one has two); D5-D7 chain 2-6 goals and need up to 1650 steps (Sec. III-D) | task specs, `step_budget` | `python scripts/export_task_table.py` | SUPPORTED |
| C-08 | Static checks: names, orderings, waypoints (Sec. III-D) | `afford_pro/bench/tasks/__init__.py:static_problems` | `pytest tests/test_bench_tasks.py -k spec_problems` | SUPPORTED |

## Simulation, perturbations, and control

| ID | Claim (section) | Artifact | Status |
| --- | --- | --- | --- |
| C-10 | Action space, 20 Hz, 25 x 2 ms substeps, DLS IK, gripper limits (Sec. III-A) | `afford_pro/envs/base_franka_env.py` | SUPPORTED |
| C-11 | Step budget 400 + 250 (k - 1) (Sec. III-A) | `afford_pro/bench/spec.py:step_budget` | SUPPORTED |
| C-12 | L1: per-object regions (half-widths 2-40 mm, yaw up to +-10 to +-45 deg), receptacles or assemblies; TCP jitter 1.5 cm; rejection sampling (Sec. IV) | scenes, task specs, `GoalFrankaEnv._layout_problem`, `PerturbationConfig` | SUPPORTED |
| C-13 | L2 7-12 cm, yaw 45-90 deg; L3 lights; color +-25% on moved objects; L4 friction 0.12-0.25 on moved objects and pads, mass x1.8-2.5; L5 joint (Sec. IV) | `afford_pro/perturbations/engine.py`, `GoalFrankaEnv._perturb_bodies` | SUPPORTED |
| C-14 | Suites are paired on seeds; parameters restored each reset (Sec. IV) | `tests/test_perturbation_isolation.py` | SUPPORTED |
| C-15 | Physics: gravity-compensated servos, tendon gripper, pads 4 ms with torsional friction, elliptic cones impratio 10, objects without torsional friction, native convex multi-point contact (Sec. IV) | scene XMLs, `scripts/build_bench_scenes.py`, `tests/test_bench_tasks.py::test_objects_touch_each_other_without_torsional_friction` | SUPPORTED |
| C-16 | Oracle: grasp planning, carry, re-centering, vertical empty-hand approaches (Sec. V-A) | `afford_pro/bench/oracle.py`, `afford_pro/oracles/parametric_oracle.py`, travel test | SUPPORTED |
| C-17 | Demonstrations: 50 per task, schema, provenance, validator, exact replay (Sec. V-B) | `afford_pro/datasets/`, `tests/test_bench_tasks.py::test_dataset_is_written_validated_and_replayed_exactly`, pilot files | PARTIALLY SUPPORTED: the pipeline is implemented and tested; the 10,500 demonstrations are not generated yet (D9) |
| C-18 | Converters to RLDS and LeRobot (Sec. V-B) | `tests/test_converters.py` | SUPPORTED |
| C-19 | Metrics: SR with task bootstrap, goal completion, PRI, GDI, force violation rate (Sec. VI-C) | `scripts/evaluate_policy_suite.py`, `afford_pro/evaluation/` | SUPPORTED |

## Results

| ID | Claim (section) | Evidence | Status |
| --- | --- | --- | --- |
| C-20 | Oracle 2096/2100 nominal (99.8%), every task >= 9/10; random and null 0/420 each (abstract, Sec. VII-B) | `task_validation_2026-10-19_nominal.json` (E-016) | SUPPORTED |
| C-21 | 4,200 resets: no goal at start, no fallback; region starts s.d. >= 0.99 cm (Sec. IV, VII-B) | same file | SUPPORTED |
| C-22 | Oracle 98.9/97.5/95.5% under L4/L2/L5 (Sec. VII-B) | `task_validation_2026-10-19_{p4_physical,p2_spatial,combined_ood}.json` (E-017) | SUPPORTED |
| C-23 | Table II: reference policies per domain and suite | `leaderboard_2026-10-19/` (E-019) | SUPPORTED |
| C-24 | 63,000 rollouts per policy for 50 episodes x 210 tasks x 6 suites (Sec. VIII-A) | arithmetic | SUPPORTED |
| C-25 | H1-H3, hardware pilot, Stage 2 charter | plans | NOT APPLICABLE |

## Related work (sources in `LITERATURE_COMPARISON.md`)

| ID | Claim | Status |
| --- | --- | --- |
| C-30 | LIBERO-PRO: >90% to near 0% when initial positions or task goals change; robust to appearance and paraphrase | SUPPORTED (repo leaderboard and paper, per literature audit) |
| C-31 | LIBERO-X: 39.4% average over five VLAs at Level 1 | SUPPORTED (verified in `scratch/libero_x_raw.txt`) |
| C-32 | LIBERO's On allows 3 cm | SUPPORTED (verified in `../LIBERO-PRO/libero/libero/envs/object_states/base_object_states.py`) |
| C-33 | LIBERO-Plus: seven factors, 10,030 variants, five difficulty levels | SUPPORTED (literature audit, paper sections cited) |
| C-34 | COLOSSEUM: 14 factors, friction and mass on supporting tasks, 30-50% single and >= 75% combined | SUPPORTED |
| C-35 | ManiSkill3 stacking requires release and rest | SUPPORTED (code checked by the literature audit) |
| C-36 | Factory simulates for RL; IndustReal transfers to a real Franka | SUPPORTED |
| C-37 | Bibliography entries | SUPPORTED after the 2026-10-17 corrections (arXiv API metadata for 10 entries); Open X-Embodiment author list and CQL pages UNVERIFIED |
| C-38 | LIBERO writes each task's goal as a conjunction of BDDL predicates | SUPPORTED (`../LIBERO-PRO/libero/libero/bddl_files/*/*.bddl`, e.g. `(:goal (And (Turnon flat_stove_1) (On moka_pot_1 ...)))`) |

## Claims removed in the 2026-10-19 rewrite
The task tiers (66 core, 35 reach, 109 mismatches), the single-body success predicate, the released single-body
dataset and its limitations, the L4 friction x mass sweep, the oracle force-violation rate and the one-task pipeline
check describe the deprecated legacy task set; their evidence (E-002 to E-015) stays in this folder.
