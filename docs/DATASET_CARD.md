# AFFORD-Bench nominal demonstrations: dataset card

Status: not yet hosted for download. Licence: MIT (top-level `LICENSE`); robot meshes Apache-2.0.

## Contents
- 210 files `datasets/<task_id>_nominal.hdf5`, one per registered task (tasks 1-210), 50 demonstrations each:
  10,500 demonstrations, 1,230,842 control steps (mean 117.2, median 119, max 600), about 447 GB.
- Per-file sizes, status, and step counts: `docs/roadmap/evidence/dataset_validation_2026-10-17/manifest.json`.
- Use the task tiers (`afford_pro.task_tiers`): only the 66 `core` tasks have instructions that match the simulated
  scene and success predicate; 35 `reach` tasks and 109 `instruction_mismatch` tasks are documented in
  `docs/roadmap/AUDIT_REPORT.md` (A-04, A-20).

## Collection
- Robot: simulated Franka Panda (MuJoCo Menagerie meshes, modified model; `afford_pro/assets/robots/NOTICE.md`),
  MuJoCo 3.5.0, 20 Hz control (25 x 2 ms physics steps), Cartesian position increments with yaw, force-limited gripper.
- Demonstrator: privileged closed-loop scripted oracle (`afford_pro/oracles/parametric_oracle.py`) that reads the
  object pose; only successful episodes were kept (about 5% of sampled starts were discarded, inferred from seed
  gaps), so the start distribution of the data is conditioned on oracle success.
- Starts: L1 nominal scatter (`afford_pro/perturbations/engine.py`); in 29 tasks the start varies by less than 1 cm.
- Seeds: per-demo attribute `seed`; at least three generation runs with different seed bases (A-06).

## Schema (Robomimic HDF5; see README "Dataset schema")
`data/demo_i/{actions (T,7) float32 in [-1,1], rewards, dones, states (T,85) float64, obs/*, next_obs/*}`; images
`obs/agentic_rgb` (T,640,660,3) and `obs/wrist_rgb` (T,300,420,3) uint8; proprioception; robomimic/LIBERO aliases;
`mask/{train,valid}` 45/5 per file; root attributes `env_args`, `problem_info`, `model_file`, camera intrinsics/poses.
Rewards and dones are 0 except 1 at the last step (the first step at which success holds). No depth.

## Validation
Every episode was checked by `scripts/validate_datasets.py` (E-007): 0 errors; warnings listed below. 630 demos were
replayed from their recorded initial state: 595 reproduce the recorded success step exactly.

## Known limitations
1. No provenance attributes (code version, simulator version, generation command); the working tree was dirty.
2. Recorded actions are float32 roundings of the float64 actions that were executed, so replays are close but not
   exact (median object deviation 0.02 mm; 35 of 630 replays change the success step). Fixed in the generator.
3. The state after the last action is not stored; `next_obs[T-1]` repeats `obs[T-1]` (offline RL: wrong terminal
   transition). Fixed in the generator.
4. 114 demos are longer than the environment's 500-step limit; 130 longer than the evaluation budget of 400 steps.
5. `language_instruction` rotates over four variants per file; the metric variant gives nominal, not actual, start
   coordinates. Use `data.attrs['instruction']` (canonical) unless studying paraphrases.
6. 6 files store observation aliases as copies (larger files, identical content).
7. Instructions of non-core tasks describe objects or steps that are not simulated.

## Intended use
Imitation learning and offline RL on the core tier, with evaluation through `scripts/evaluate_policy_suite.py`
(seeds disjoint from the generation seeds of each task). Not suitable for language-grounding claims on non-core
tasks or for sim-to-real claims without the hardware study in the paper's Stage 2 charter.
