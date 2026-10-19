# AFFORD-Bench demonstrations: dataset card

Status: generated with `scripts/generate_all_150_tasks.py`; not yet hosted for download. Licence: MIT (top-level
`LICENSE`); robot meshes Apache-2.0.

## Contents
- 210 files `datasets/<task_id>_nominal.hdf5`, one per task, 50 demonstrations each (10,500 in total).
- Every demonstration completes all goals of its task, in order, within the task's step budget (400 steps plus 250 per
  further goal), from a start sampled by the L1 nominal suite.

## Collection
- Robot: simulated Franka Panda (MuJoCo Menagerie meshes, modified model; `afford_pro/assets/robots/NOTICE.md`),
  MuJoCo 3.5.0, 20 Hz control (25 x 2 ms physics steps), Cartesian position increments with yaw, force-limited gripper.
- Demonstrator: privileged closed-loop scripted oracle (`afford_pro/bench/oracle.py`) that reads the object poses.
  Only episodes that complete the task are kept, so the start distribution of the data is conditioned on oracle
  success; starts that fell back to a default layout are never kept. Rejected seeds are recorded in the file.
- Seeds: `42 + 1000 * task_number + k` (disjoint from the evaluation block `200000 + 1000 * task_number + j`); per-demo
  attribute `seed`.

## Schema (Robomimic HDF5)
`data/demo_i/{actions (T,7) float32 in [-1,1], rewards, dones, states (T,n) float64, obs/*, next_obs/*}` with the
attributes `seed`, `init_state` and `final_state` (the state after the last action);
images `obs/agentic_rgb` (T,640,660,3) and `obs/wrist_rgb` (T,300,420,3) uint8; proprioception
(`joint_positions`, `joint_velocities`, `tcp_pose`, `gripper_width`); robomimic/LIBERO aliases; `next_obs[T-1]` is
the observation after the last action; `mask/{train,valid}` 45/5 per file. Root attributes: `env_args` (with the task
set, so the environment can be rebuilt), `problem_info`, `model_file`, camera intrinsics and poses, the canonical
instruction and its step-by-step variant (`prompt_variants`), and `provenance` (code commit, simulator and package
versions, asset hashes, command). Rewards and dones are 0 except 1 at the last step (the first step at which every
goal has been reached and holds). No depth.

## Validation
`scripts/validate_datasets.py` checks every episode (shapes, finite values, terminal reward and done, alignment of
observations with recorded states, duplicates, splits, episode length against the task's budget) and replays sampled
demonstrations of every file from their recorded initial states: every recorded generalized coordinate must be
reproduced exactly.

## Intended use
Imitation learning and offline RL, with evaluation through `scripts/evaluate_policy_suite.py` on the six paired suites
(seeds disjoint from the generation seeds). Not suitable for sim-to-real claims without the hardware study in the
paper's Stage 2 charter.

## Deprecated files
The earlier single-body task set's 2026-10 files (210 files, 460.6 GB, plus a 7-file archive, 18 GB) were deleted on
2026-10-20 at the authors' request; they were never hosted. The generator refuses to overwrite a file of another task
set, so any copy of them must be kept outside `datasets/`.
