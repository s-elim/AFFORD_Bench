# Claim-evidence ledger

Every quantitative or scientific claim in `AFFORD-Bench_RSS'27/paper-template-latex/main.tex` as revised on
2026-10-17, with its artifact. Status: SUPPORTED, PARTIALLY SUPPORTED, UNVERIFIED, CONTRADICTED, NOT APPLICABLE
(design specification or plan, not a result). Evidence paths are relative to `docs/roadmap/evidence/`; E-xxx refer
to `EXPERIMENT_LOG.md`. All runs: MuJoCo 3.5.0 on the dirty tree described in each provenance sidecar.

The previous ledger (claims of the pre-revision paper, several CONTRADICTED) is summarized at the end.

## Task inventory and data

| ID | Claim (section) | Artifact | Reproduce | Status |
| --- | --- | --- | --- | --- |
| C-01 | 210 registered tasks in 7 domains, 30 each (abstract, Sec. III-C) | `afford_pro.list_tasks()` | `pytest tests/test_task_registry.py` | SUPPORTED |
| C-02 | 66 core, 35 reach, 109 instruction mismatches (66 absent referent, 31 partial chain predicates, 12 audited); per-domain 17/6/7, 20/7/3, 17/8/5, 12/14/4, 0/0/30 (abstract, Table I, Sec. III-C) | `afford_pro/task_tiers.py`, `task_audit_2026-10-17.json` | `python -c "from afford_pro.task_tiers import tier_counts_by_domain; print(tier_counts_by_domain())"` | SUPPORTED (tiers are defined by the audit; the 12 audited reasons are in `AUDITED_MISMATCHES`) |
| C-03 | 17 core goals on unmarked surfaces; 13 core tasks with near-fixed starts (Sec. III-C) | `task_audit_2026-10-17.json` | E-005 | SUPPORTED |
| C-04 | 57 of 90 D5-D7 tasks end before the last instructed step (Sec. III-C) | `configs/prompts_augmented.yaml` | count `episode_terminates_after_subtask < len(atomic_subtasks)` | SUPPORTED |
| C-05 | 10,500 demonstrations, 1,230,842 steps, mean 117 (abstract, Sec. V-B) | `dataset_validation_2026-10-17/summary.md` | E-007 | SUPPORTED |
| C-06 | Dual RGB 640x660 and 300x420; proprioception; full simulator state; intrinsics and poses stored; depth not in released data (Sec. III-B, V-B) | E-007 report | E-007 | SUPPORTED |
| C-07 | 90/10 train/validation masks; every demo from its own seed with a distinct initial state (Sec. V-B) | E-007 (no duplicate_initial_state errors; split stats) | E-007 | SUPPORTED |
| C-08 | About 5% of sampled starts were discarded (Sec. V-B) | E-002 seed gaps (597/11,097 = 5.4%) | `seedscan.py` (scratch) | SUPPORTED (inferred from seed gaps; generator did not log failures) |
| C-09 | Validator finds no errors in any episode (Sec. V-B) | `dataset_validation_2026-10-17/summary.md` | E-007 | SUPPORTED |
| C-10 | Replay: 595/630 exact success step, median deviation 0.02 mm, 35 differ (Sec. V-B) | `dataset_validation_2026-10-17/report.json` | E-007 | SUPPORTED |
| C-11 | 114 demos exceed the 500-step env limit; final state not stored; three generator invocations (Sec. V-B) | E-002, E-007 | E-007 | SUPPORTED ("at least three" seed schemes) |
| C-12 | Four instruction variants per task, annotated with whether the simulator evaluates them (Sec. I) | `configs/prompts_augmented.yaml` | `pytest tests/test_prompts_augmented.py` | PARTIALLY SUPPORTED: annotations are too generous for reach tasks (the tiers correct this) |
| C-13 | Converters export LeRobot v2.0 and RLDS (round-trip tests on synthetic episodes; RLDS NumPy layout) (Sec. VIII-A) | `tests/test_converters.py` | pytest | SUPPORTED as stated |

## Simulation, perturbations, and control

| ID | Claim (section) | Artifact | Reproduce | Status |
| --- | --- | --- | --- | --- |
| C-14 | Frictional grasping; no attachment, snapping, or teleport; success predicate as in Eq. (1) (abstract, Sec. III-A) | `afford_pro/envs/parametric_env.py` | code inspection | SUPPORTED for the 210 core-registry tasks (frontier envs not audited) |
| C-15 | Tolerance tiers 2-2.5 mm / 5 deg and 8 mm / 15 deg (abstract, Table I) | `task_definitions.PRECISION_TIERS` | `pytest tests/test_task_registry.py` | SUPPORTED |
| C-16 | Action and controller parameters (0.025 m, 0.5 m/s, 20 Hz, yaw 0.05 rad, about 35 N per finger) (Sec. III-A) | `base_franka_env.py`, scene XML actuator8 (70 N tendon, 0.5/0.5 split) | code inspection | SUPPORTED |
| C-17 | L1 scatter and shrink schedule 1/2, 1/4, 1/10, 1/20; 29/210 tasks with start std < 1 cm; task 12 always nominal; distractors closer than 9 cm in 7/76 tasks, min 3.9 cm (Sec. IV) | `perturbations/engine.py`, `parametric_env.RESET_SCATTER_SCHEDULE`, E-005 | E-005 | SUPPORTED |
| C-18 | L2: radius U(7, 12) cm, yaw U(45, 90) deg (Sec. IV) | `PerturbationConfig` | code | SUPPORTED |
| C-19 | L3: diffuse x U(0.5, 1.5) clipped at 1; key light 0.9 so at most x1.11 (Sec. IV) | `apply_visual_perturbation`, scene XML lights | code | SUPPORTED |
| C-20 | L4: mu U(0.12, 0.25) on body and pads, mass/inertia x U(1.8, 2.5) (Sec. IV) | `apply_physical_perturbation` | code | SUPPORTED |
| C-21 | Perturbations restored and seeded per episode (Sec. IV) | `PerturbationEngine.begin_episode` | code | SUPPORTED; isolation test (each mode changes only its factor) still to write (T5) |
| C-22 | L4 sweep: >= 94% for mu >= 0.12 up to x2.5; 70% at (0.12, x5), 38% at (0.05, x5), 16% at (0.02, x5) (Sec. IV, abstract) | `l4_sweep_2026-10-18.csv` | E-013 | SUPPORTED (66 core tasks, 3 seeds per cell) |
| C-23 | Force: median 0 N, p95 143 N, max 169 N; 10/210 > 150 N; oracle FVR 4.8% (Sec. III-A) | `force_probe_2026-10-18.csv` | E-013 | SUPPORTED |

## Results

| ID | Claim (section) | Artifact | Reproduce | Status |
| --- | --- | --- | --- | --- |
| C-24 | Oracle 95.7% nominal (1005/1050), 94.5% within H = 400, L4 95.0%; random 0/1050; null 0/210; no success at reset; per-domain and core rows of Table II with task-bootstrap CIs (abstract, Table II, Sec. VII-B) | `sanity_2026-10-18/summary.{md,json}` | E-013 | SUPPORTED |
| C-25 | 28 tasks fail at least once; task 95 fails 5 of 5, a chain task 4 of 5 (Sec. VII-B) | `sanity_2026-10-18/episodes_*.csv` | E-013 | SUPPORTED |
| C-26 | Low-dim BC on task 61 solved 16/20 (pipeline check, not a baseline) (Sec. VII-B) | `pipeline_check_2026-10-17/bc/`, checkpoint sha256 7e9d4dd9... | E-011 | SUPPORTED (single task, single seed; not a baseline result) |
| C-27 | Reactive probe: closed loop and 16-step chunks 14/18 each, open loop 0/18 (Sec. VIII-B) | `reactive_perturbation_2026-10-17.csv` | E-010 | SUPPORTED |
| C-28 | 16,500 rollouts per policy, about 140 robot-hours at 2 trials/min (Sec. VIII-A) | arithmetic 50 x 66 x 5; 8,250 min | - | SUPPORTED |
| C-29 | H1-H3 (Sec. VII-A); Stage 2 charter (Sec. VIII-B) | - | - | NOT APPLICABLE (hypotheses and plans) |

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

## Pre-revision claims that were removed or corrected
99.8% oracle and 47.3% L4 (CONTRADICTED, A-01/A-18); "MuJoCo 3.2.7" (CONTRADICTED); 108 steps and 1.1e6 timesteps
(CONTRADICTED); F_max rationale (CONTRADICTED, A-28); reactive 18/19 and 9/9 (stale, re-measured); slip-probe
paragraph (stale, removed); LIBERO-X ">90% to near 0%" and Factory real transfer (CONTRADICTED, A-24); "35 suites, all
implemented" as a headline (kept only as a design statement); README perturbation ranges and dataset note
(CONTRADICTED, A-10, README rewrite pending); synthetic VLA/IL scores (A-02, quarantined).
