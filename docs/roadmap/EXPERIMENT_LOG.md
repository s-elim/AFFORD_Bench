# Experiment log

Each entry: command, configuration, versions, seeds, outcome, compute. An entry is marked successful only when its
output was inspected against its success criterion. Environment unless stated: Windows 11, Ryzen 9 3900X, Python
3.13.5, MuJoCo 3.5.0, numpy 2.2.6, h5py 3.12.1, torch 2.10.0+cpu; checkout `main` @ `cba995e` + uncommitted changes
(dirty tree; the provenance sidecars list the modified files and their hashes).

## E-001 Baseline unit tests (2026-10-17)
- Command: `python -m pytest -q tests -p no:cacheprovider`
- Outcome: 1722 passed in 120 s. Success criterion (all pass): met.

## E-002 Dataset low-dim scan, all 210 nominal files (2026-10-17)
- Commands: scratch scripts `explore_ds.py` (4 files, full low-dim per episode) and `seedscan.py` (all files: seeds,
  lengths, alias storage, model_file). Read-only. About 5 min.
- Outcome: 210 files, all registered task ids, 50 demos each (10,500). Seed schemes: 202 files `42+1000n+k`, 7 files
  `42+100n+k`, 1 file `42+k`. Inferred attempts 11,097, failures 597 (94.6% oracle success during generation; per
  domain D1 93.6, D2 97.3, D3 98.0, D4 90.6, D5 96.2, D6 90.4, D7 97.0%; worst `lab_micro_tube_to_rack_b` 50/164).
  130 demos > 400 steps, 114 > 500 steps. 6 files store aliases as copies, 204 as hard links. All model_file
  attributes contain `impratio` (generated after the 2026-10-16 XML change). In the 4 fully scanned files: unique
  seeds and initial states, `init_state == states[0]` for all demos, rewards and dones 0 then 1 at the last step,
  actions finite in [-1, 1], instruction variants rotate 13/13/12/12.
- Caveat: attempts are inferred from seed gaps; the generator does not record failures.

## E-003 Sanity-harness timing (2026-10-17)
- Command: `python scripts/sanity_baselines.py --tasks industrial_shaft_assembly,kitchen_press_brew_button,chain_burger_flip_cheese_top`
- Outcome: 77 s for 3 tasks (about 26 s per task with 5+5 oracle, 5 random, 1 null episodes). Oracle nominal 5/5, 5/5,
  4/5; L4 4/5, 5/5, 5/5; random and null 0. Used only for cost estimation.

## E-004 Sanity floors on the current physics, all 210 tasks (2026-10-17, done)
- Command: 8 shards of `python -I scripts/sanity_baselines.py --shard i/8 --oracle-episodes 5 --episodes 5 --horizon 400 --seed 10000 --out docs/roadmap/evidence/sanity_2026-10-17/part_i.csv --episodes-out .../episodes_i.csv`;
  aggregation `python scripts/aggregate_sanity.py --dir docs/roadmap/evidence/sanity_2026-10-17`.
- Seeds: reset seeds 10000 to 10004 per task (disjoint from all generation seed schemes found in E-002).
- Compute: about 15 min wall clock, 8 processes.
- Outcome (`summary.md`, `summary.json`; intervals are 95% task-clustered bootstrap): oracle nominal 95.3%
  (93.4 to 97.0) within its 600-step budget, 94.6% within H = 400; oracle L4 95.9% (94.4 to 97.3); random 0/1050;
  null 0/210; no task succeeds at reset. Per domain (nominal): D1 95.3, D2 97.3, D3 100, D4 94.0, D5 93.3, D6 92.0,
  D7 95.3. Core tier 95.2 (91.2 to 98.2), reach tier 100. Successful oracle episodes: mean 115.2 steps, median 120.
- Interpretation: contradicts the paper's 99.8% nominal and 47.3% L4 (measured 2026-10-08 on MuJoCo 3.2.7 with
  pyramidal friction cones). Under the current elliptic cones, the L4 perturbation does not reduce oracle success.
- Success criterion (complete run, every shard wrote its CSV, provenance recorded): met.

## E-005 Runtime task audit, all 210 tasks (2026-10-17, done)
- Command: `python -I scripts/audit_tasks.py --resets 10 --workers 4 --out docs/roadmap/evidence/task_audit_2026-10-17.json`
- Seeds: 300000 to 300009 (disjoint from generation and evaluation blocks). About 10 min.
- Outcome: no success at reset; start fallback only in task 12 (always) and 12 tasks at 10 to 30%; hold goals 74,
  93, 160, 172 (74 is a "place" instruction); 29 tasks (14 core by the prompt flags) have start xy std < 1 cm;
  24 of the 78 flag-core tasks place onto an unmarked surface point (table, bench, board, workstation); distractors
  closer than 9 cm to start or goal in 7 of 76 tasks (minimum 3.9 cm, task 29). 12 flag-core tasks re-classified as
  instruction mismatches after reading the audit (`afford_pro/task_tiers.AUDITED_MISMATCHES`): core = 66.

## E-007 Episode-level validation of all 210 datasets with replay (2026-10-17, done)
- Command: `python -I scripts/validate_datasets.py --replay-per-file 3 --workers 8 --out-dir docs/roadmap/evidence/dataset_validation_2026-10-17`
- Scope: every episode of every file (low-dim arrays, terminal semantics, obs/state alignment, next_obs shift,
  duplicates, masks, registry); image shape/dtype for all, content (dark/constant) on the first and last frame of
  every episode; replay of 3 demos per file (first, middle, last). About 1.5 h with 8 workers.
- Outcome (`summary.md`, `report.json`, `manifest.json`): 210 files, 10,500 demos, 1,230,842 steps (mean 117.2);
  0 INVALID, 210 VALID_WITH_WARNINGS. Warnings: no provenance (210 files), terminal next_obs copies obs (all
  10,500 demos), aliases stored as copies (6 files), demos over the 500-step env limit (12 files) and over H = 400
  (20 files). No duplicate initial states or action sequences within or across files.
- Replay: 595/630 demos reach success exactly at their recorded last step; 35 differ (20 succeed 1-32 steps early,
  15 do not succeed by T). Object deviation median 2.3e-5 m, 90th percentile 2.7e-4 m, max 0.085 m (10 > 1 mm).
- Diagnostics (scratch scripts `warmstart_diag.py`, `divergence_diag.py`, `warm_every_step.py`, fresh oracle
  episodes on 4 tasks): replay diverges at the first action by about 1e-9 in free-body velocities; the solver warm
  start before step 0 matches exactly, and imposing the recorded warm start before every step does not remove the
  divergence. Root cause: open (not the warm start).

## E-008 L4 friction x mass sweep on the 66 core tasks (2026-10-17, done)
- Command: `python -I scripts/l4_sweep.py --tier core --mu 0.02,0.05,0.12,0.25 --mass 1.0,2.5,5.0 --seeds 3 --workers 6 --out docs/roadmap/evidence/l4_sweep_2026-10-17.csv`
- Seeds 400000 to 400002. 2,574 oracle episodes, about 25 min.
- Outcome (oracle success %, task-clustered 95% CI): nominal 95.5 (90.4-99.5); mu 0.25: x1 96.0, x2.5 94.4,
  x5 91.9; mu 0.12: 96.0, 91.4, 71.2; mu 0.05: 88.4, 59.6, 36.9; mu 0.02: 51.5, 31.8, 14.1.
- Interpretation: the current L4 range (mu 0.12-0.25, mass x1.8-2.5) lies on the flat part of the curve; the
  physical axis lowers expert success only for mu <= 0.05 or heavy (x5) bodies.

## E-009 Oracle contact-force probe (2026-10-17, done)
- `python -I scripts/force_probe.py --episodes 1 --workers 6 --out docs/roadmap/evidence/force_probe_2026-10-17.csv` (seed 500000).
- Outcome: 203/210 successful; peak robot-scene force median 0 N, 95th percentile 142.6 N, max 163.2 N; 9 of 210
  episodes exceed F_max = 150 N (all D7 kitchen chains).

## E-010 Reactive perturbation probe on the current physics (2026-10-17, done)
- `python -I scripts/test_reactive_perturbation.py --out docs/roadmap/evidence/reactive_perturbation_2026-10-17.csv`
  (defaults: 3 tasks, 3 seeds, T = 60, H = 16, 4 cm displacement).
- Outcome: closed loop 14/18, 16-step chunks 14/18, open-loop replay 0/18. No undisplaced control in this script
  version (the paper's earlier "9 of 9 without displacement" has no current artifact and was removed).

## E-011 Baseline pipeline validation, one task (2026-10-17, done)
- `scripts/train_policy.py --policy {bc,diffusion} --modality lowdim --task tabletop_sort_red --epochs 200 --seed 0`
  then `scripts/evaluate_policy_suite.py --checkpoint ... --episodes 20` (seeds 261000-261019). Checkpoints and
  logs in the session scratchpad (not evidence for the paper's baselines).
- Outcome: BC 16/20 (Wilson 58-92%), best validation L1 loss 0.021; diffusion 0/20, best validation loss 0.088.
  The in-repo diffusion implementation does not learn this task at a budget where BC does; use a reference
  implementation (LeRobot 0.5.1 is installed) before reporting a diffusion baseline.

## E-012 Replay-divergence root cause (2026-10-18)
- Scratch diagnostics `mjdata_diff.py` (diff of every MjData array before the first action, original vs replay) and
  `divergence_diag.py` on fresh oracle episodes of 4 tasks.
- Outcome: before the fix, only uninitialized sparse-index scratch arrays (efc_AR_rowadr, efc_AR_rownnz,
  efc_J_colind) differ, yet the first step diverges by about 1e-9: the executed action was float64 and the recorded
  one float32. After `BaseOracle.act` executes the float32 action, all 4 replays are bit-identical (118, 127, 112,
  104 steps). Regression test `tests/test_replay.py`.

## E-013 Re-measurement after the seed-pairing and float32-action fixes (2026-10-18, done)
- Same commands as E-004, E-008, E-009, with outputs `sanity_2026-10-18/`, `l4_sweep_2026-10-18.csv`,
  `force_probe_2026-10-18.csv`. Supersedes E-004/E-008/E-009 for the paper and the page.
- Sanity: oracle nominal 95.7% (1005/1050, task-bootstrap 93.8-97.4), 94.5% within H = 400, L4 95.0% (93.0-96.8);
  core tier 95.2 / 93.3 / 96.1%; random 0/1050, null 0/210, no success at reset; 28 tasks fail at least once
  (`lab_micro_tube_to_rack_b` 5/5 failures).
- Sweep (core, 3 seeds): >= 94% for mu >= 0.12 up to x2.5; (0.12, x5) 69.7%, (0.05, x5) 38.4%, (0.02, x5) 15.7%.
- Force: median 0 N, p95 142.6 N, max 168.7 N, 10/210 episodes > 150 N (4.8%).

## E-014 End-to-end check of the fixed generation pipeline (2026-10-18)
- `python scripts/generate_demos.py --task tabletop_sort_red --num-demos 3 --output-dir <scratchpad>/newgen` (real
  rendering), then `python scripts/validate_datasets.py --files .../tabletop_sort_red_nominal.hdf5 --replay-per-file 3
  --expected-demos 3`.
- Outcome: status written, 3 demos in 3 attempts, exit 0; validator VALID with no issues (provenance present, final
  observation stored, canonical instructions); all 3 replays exact (object and joint deviation 0.0). The validator now
  treats any replay deviation as an error for files that carry provenance.

## E-006 Generator and validator unit tests (2026-10-17)
- `python -m pytest -q tests/test_dataset_validation.py tests/test_generation.py tests/test_task_registry.py tests/test_dataset_recording.py tests/test_converters.py`
- Outcome: all pass (validator 15, generation 4, registry 638, recording + converters 13).
