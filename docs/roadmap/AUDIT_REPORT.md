# AFFORD-Bench audit report

Started 2026-10-17. Checkout: `main` @ `cba995e` with 56 modified tracked files and untracked `data/init_states/*.npz` (180)
and `datasets/archive_oct03_audit_evidence/` (7 HDF5, 18 GB). Machine: Windows 11, Ryzen 9 3900X (12C/24T), 64 GB RAM,
GTX 1080 Ti (unused: torch is the CPU build), Python 3.13.5, MuJoCo 3.5.0, h5py 3.12.1, torch 2.10.0+cpu.

Every finding below states its evidence. "Confirmed" means reproduced on this checkout. Hypotheses and items needing
experiments are listed separately. Severity: CRITICAL (a paper claim is false or unsupported as stated, or the data
cannot be trusted for its stated use), HIGH (a scientific conclusion would be wrong or unfalsifiable), MEDIUM
(reproducibility or clarity), LOW (hygiene).

Earlier audits (`docs/reviews/rss27_stage1_audit_2026-10-08.md`, `docs/reviews/reality_check_hardening_2026-10-08.md`)
were read as history, not as evidence; each claim taken from them is re-checked here.

## Confirmed findings

### A-01 CRITICAL: Table II and the abstract report numbers measured on different physics
- Claim (main.tex abstract, Table II, Sec. VII-B): oracle 99.8% nominal (1048/1050), 47.3% L4, null/random 0%,
  "All results use one simulator build (MuJoCo 3.2.7)".
- Evidence: `docs/reviews/sanity_baselines_2026-10-08.csv` predates the uncommitted 2026-10-16 change of every scene
  XML to `cone="elliptic" impratio="10"` (`git diff afford_pro/assets`), and the installed simulator is MuJoCo 3.5.0.
  The released datasets were generated with that changed physics (all 210 `model_file` attributes contain `impratio`).
- Generation-time evidence on the current physics: seed gaps in the 210 datasets imply 10,500 successes in about
  11,097 oracle attempts (94.6%; 90.4% to 98.0% per domain; `lab_micro_tube_to_rack_b` about 30%). Inference assumes
  consecutive attempt seeds (true for the current generator).
- Remediation: re-measured with `scripts/sanity_baselines.py` (E-004); paper numbers must come from that run.

### A-02 CRITICAL: synthetic "model results" remain in the repository
- `scripts/evaluate_vla_baselines.py` holds hardcoded score dictionaries (BENCHMARK_PROFILES) for pi0, SpatialVLA,
  AtomVLA, CoT-VLA, DP, ACT, OpenVLA, SmolVLA. Their outputs are in `docs/reviews/vla_model_benchmark_comparison_2026-10-08.{csv,json}`,
  `docs/MODEL_BENCHMARK_EVALUATION.md`, and the figure `figures/model_performance_comparison.{pdf,png}` (also
  `docs/images/`). A guard flag and a disclosure banner were added on 2026-10-16, but the files still read as results.
- Remediation: quarantine under `docs/quarantine/` with an explicit "not experimental results" header; never cite.

### A-03 HIGH: the benchmark-comparison figure characterizes other benchmarks without evidence
- `figures/benchmark_comparison.png` (from `scripts/generate_benchmark_comparison_figure.py`) states LIBERO-PRO and
  LIBERO-X use "Coarse / Kinematic Welds", "Nominal (Fixed mu = 0.5)", "VLA Models Only", "Unverified Null Baselines",
  and "1x RGB (128x128)". LIBERO is built on robosuite, which grasps through contact; no source for these cells is
  in the repository. Verdicts per cell: `LITERATURE_COMPARISON.md`.
- The figure is not referenced by `main.tex` but is on the project page (`docs/index.html`, `docs/images/`).

### A-04 HIGH: task instructions and success predicates disagree for most of the benchmark
- `configs/prompts_augmented.yaml` (machine-checked against `task_definitions.py` by `tests/test_prompts_augmented.py`):
  112 of 210 tasks have the named object in the scene and a success predicate covering the instruction;
  67 name an object that is not simulated (e.g. task 4 "steel ball bearing ring" is the spur gear, task 5 "flange
  bolt" is the dowel pin); 31 more are chains whose episode ends when the single evaluated body reaches its goal.
  By domain, D5 0/30, D6 0/30, D7 0/30 tasks have a covering predicate.
- 35 tasks (`task_type="press_button"`) succeed when the TCP stays within the tier tolerance of a fixed world point
  for 5 steps. Their instructions promise actions that are not simulated: "wipe sweep" (58, 59, 89, 109, 110),
  "press ... to dispense reagent" (107), "verify threaded seal" (117), "press down gently to settle layers" (50),
  "press the START button" (99). No button, wipe, or seal mechanism exists, and the goal point does not move with
  the scene, so these tasks are solvable by memorizing one world coordinate. `prompts_augmented.yaml` marks them as
  covered by success; that flag is too generous for this type.
- Consequence: language-grounding, H3 (chain compounding), and "210 manipulation tasks" claims are not supported
  as stated. See ROADMAP T3 for the task tiering.

### A-05 HIGH: the "state" baselines cannot see the object
- `scripts/train_policy.py` state modality = `[tcp_pose(7), joint_positions(7), joint_velocities(7), gripper_width(1)]`.
  The object pose is not an input and is not stored in `obs/` (only inside `states`). With L1 scatter of +-7 cm and
  +-45 deg, a state-only policy is blind to the target; its success would measure trajectory averaging.
- Also: no input or action normalization; checkpoint "best" is chosen by training loss (the `mask/valid` split is
  never used); diffusion sampling at evaluation is unseeded; visual input is the workcell image subsampled to
  80x80; `evaluate_id.py` reports no confidence interval and runs 10 episodes by default (paper says 50).
- Existing checkpoints (`checkpoints/*.pt`) are 1 to 10 epoch CPU smoke runs on single tasks with no evaluation
  record or dataset hash. They are not evidence of learning.

### A-06 HIGH: dataset provenance is incomplete and the corpus mixes generation runs
- Stored: task id, instruction(s), env_args (perturbation mode, depth flag, "env_version 0.1.0"), scene XML, camera
  K and poses, per-demo seed and init state, full physics states. Not stored: git commit or dirty state, simulator
  and package versions, oracle configuration, generation command, timestamps, attempt and failure counts.
- At least three invocations produced the corpus: 7 files use episode seeds `42 + 100*n + k`
  (`industrial_shaft_assembly`, `kitchen_patty_stack`, `tabletop_sort_red`, `lab_transfer_test_tube`,
  `long_horizon_transmission_assembly`, `chain_shaft_bearing_clamp`, `chain_burger_flip_cheese_top`),
  `lab_micro_tube_to_rack_b` uses `42 + k` (the default of `scripts/generate_demos.py`), and 202 use `42 + 1000*n + k`
  (current `generate_all_150_tasks.py`). The generating command line exists nowhere.
- Two writer versions: 6 files written before `hdf5_writer.py` changed (2026-10-17 01:55) store observation aliases
  as copies instead of HDF5 hard links (e.g. `industrial_shaft_assembly_nominal.hdf5` is 9.5 GB vs about 3 GB).
- The working tree was dirty during generation (scripts were edited as late as 11:48 while files were written until
  12:25), so no commit reproduces the data exactly.

### A-07 MEDIUM: demonstrations run past the environment's episode limit
- `BaseOracle.max_steps = 600` and `BaseOracle.act` ignores `truncated`; `ParametricFrankaEnv.max_episode_steps = 500`
  and the evaluation horizon is H = 400. 114 of 10,500 demos are longer than 500 steps and 130 longer than 400
  (dataset scan, E-002). Under the evaluation protocol those episodes would be failures.

### A-08 MEDIUM: language labels in the datasets are partly wrong for the episode they label
- `RobomimicHDF5Writer.add_demonstration` assigns each demo one of four instruction variants in rotation
  (13/13/12/12 per task). The "geometric" variant states the nominal start coordinates (e.g. "at (0.400, 0.240,
  0.050) m"), but every episode starts from a scattered pose up to 7 cm and 45 deg away. The affordance variant
  refers to features that are not modeled (e.g. a "narrower pilot end" on the shaft). Chain-task variants describe
  steps the demo never performs (task 181: "Flip burger patty with spatula..." while one body is moved).

### A-09 MEDIUM: dataset files are written non-atomically and the completeness check is shallow
- The writer opens the final path with mode "w"; an interrupted run leaves a truncated file at the final name.
- `verify_dataset_integrity.verify_file` (used by the generator to skip finished tasks) checks shapes only in demo_0
  and terminal flags in all demos; it does not check finiteness, per-step lengths, duplicates, obs/state alignment,
  split integrity, or registry consistency. It was loosened on 2026-10-16 to also accept 128x128 images.

### A-10 MEDIUM: README contradicts the code and the data
- Perturbation ranges: README says friction mu in [0.10, 1.20], mass 0.5x to 3.0x, L2 +-6 cm and +-30 deg, light
  [0.4, 2.2] I0, clearances +-1.5 to 20 mm; code (`perturbations/engine.py`, `task_definitions.PRECISION_TIERS`)
  uses mu in [0.12, 0.25], mass 1.8x to 2.5x, L2 radius 7 to 12 cm and |yaw| 45 to 90 deg, light [0.5, 1.5] I0,
  tolerances 2, 2.5, 8 mm. README calls `p1_object` "Physical-Dynamic"; it is a color shift.
- The README "Dataset Provenance Note" says the on-disk datasets are 2026-10-03, 128x128, and fail verification.
  The files on disk are dated 2026-10-16/17 and are 640x660 / 300x420 (E-002).
- README mixes 210 and 240 tasks and 35 and 40 suites, and states "14 validated" Domain 8 tasks without an artifact.

### A-11 MEDIUM: no licence
- No LICENSE at the repository root; the vendored Franka meshes (`afford_pro/assets/robots/assets/`, MuJoCo
  Menagerie) ship without their licence or attribution; the origin of `textures/cartoon_only.png` is not recorded.
  A public release is blocked until the authors choose a licence.

### A-12 MEDIUM: simulator version is not pinned consistently
- CI installs `mujoco==3.2.7` (`requirements-ci.txt`); the datasets were generated with 3.5.0; `setup.py` requires
  `mujoco>=3.1.0`. Contact results can change across MuJoCo releases; replay determinism across versions is untested.

### A-13 MEDIUM: paper Fig. 2 is stale
- `figures/tasks_horizontal_vis.{pdf,png}` (2026-10-08) shows a black D4 wrist image (the lab wrist camera was moved
  on 2026-10-16) and overlapping column titles. The caption's depth row is fine as an illustration but the released
  datasets contain no depth (all `env_args` have `render_depth: false`).

### A-14 LOW: L3 lighting range is partly unrealized
- `apply_visual_perturbation` multiplies `light_diffuse` by U[0.5, 1.5] and clips to 1.0. The key light's nominal
  diffuse is 0.9 (all scenes checked: industrial), so its brightening is capped at x1.11. The headlight is not
  perturbed. Paper and README state [0.5, 1.5] I0.

### A-15 LOW: silent placeholder meshes
- `base_franka_env._load_model_from_xml_string` replaces any missing mesh with a 1 cm tetrahedron and any missing
  texture with a flat color, without a warning. It is reached only when `from_xml_path` fails to open a file, so the
  vendored assets are used here, but a broken checkout would silently simulate a robot with no collision geometry.

### A-16 LOW: `.gitignore` covers `datasets/*.hdf5` but not subdirectories
- `datasets/archive_oct03_audit_evidence/*.hdf5` (18 GB) is untracked but not ignored.

### A-17 LOW: docstring errors
- `BaseFrankaEnv` docstring says the gripper is "<= ~70 N per finger"; the tendon actuator is limited to 70 N and
  splits it 0.5/0.5 (about 35 N per finger, as the paper says).

### A-18 CRITICAL: the L4 physical suite no longer changes oracle outcomes
- E-004: oracle success under L4 is 95.9% (94.4 to 97.3) vs 95.3% nominal, on the same seeds. The paper's 47.3%
  came from pyramidal friction cones; with the 2026-10-16 elliptic cones (impratio 10) the force-limited grasp holds
  every evaluated body at mu 0.12 to 0.25 and mass x1.8 to 2.5.
- Consequence: the evidence that the physical axis matters is gone. H1 ("policies lose more under P_physics") may
  still hold for learned policies, but the paper can no longer show that L4 makes tasks harder for any controller.
  Options: recalibrate L4 (lower mu, heavier mass, or off-center grasps) with an oracle-degradation curve, or
  restate L4 as a robustness check whose effect on solvability is nil for the privileged expert. Author decision (D7).

### A-19 HIGH: "scattered" starts are nearly fixed for many tasks
- E-005: in 29 of 210 tasks the evaluated body's start varies by less than 1 cm (xy std 1 to 3.5 mm), because
  bodies that start in a tray, caddy, or on a spindle fall back to 1/10 or 1/20 of the scatter. 14 of them are
  core-flagged (e.g. 2, 12, 15, 18, 21, 28, 32 to 35, 41, 42, 47 to 49); task 12 always starts at the nominal pose.
  With a fixed goal these tasks admit open-loop replay; the paper's L1 description ("U(+-7 cm)") overstates it.
  Of the 66 validated core tasks, 13 have near-fixed starts (2, 12, 15, 18, 21, 28, 32, 35, 41, 42, 47, 48, 49).

### A-20 HIGH: some flag-core tasks still contradict their instruction
- E-005 + manual reading (`afford_pro/task_tiers.AUDITED_MISMATCHES`, 12 tasks): 33/34 "onto the burger stack"
  (no stack in the scene); 46 "sear reverse side" (goal = start pose, upright); 73/75 swap (goal is an unmarked
  table point named after a block that is elsewhere); 74 swap (hold goal although the instruction says place);
  87/88 "advance 10 cm" (fixed goal under a scattered start); 94/106/113/114 "hold the tip over" (success =
  pipette released and standing on the target). Validated core: 66 tasks (D1 17, D2 20, D3 17, D4 12).
- 17 of the 66 core tasks place onto an unmarked surface point (table 5, bench 6, board 4, workstation 2) at 8 mm
  or 2 mm (28, 29, 35, 47 to 49, 70 to 72, 85, 86, 93, 98, 103, 104, 111, 112); they test recall of a world
  coordinate more than perception of a target. 13 core tasks have near-fixed starts (A-19). Report both separately.

### A-21 HIGH: the package does not install outside the source tree
- Only `afford_pro/__init__.py` existed; `envs/`, `oracles/`, `datasets/`, `perturbations/` were namespace packages,
  so `setup.py`'s `find_packages()` would omit them, and no `package_data` ships the scene XML, meshes, or the
  prompts YAML (which lives outside the package). Fixed in part on 2026-10-17 (`__init__.py` added); package data
  and a wheel test are pending (T8).

### A-22 MEDIUM: the success state is not stored
- `BaseOracle.generate_episode` returns observations[:T] and states[:T]; the writer sets next_obs[T-1] = obs[T-1].
  The state after the last action (where success holds) is in no file, so success labels can be checked only by
  replay, and offline-RL loaders see a terminal transition (s, a, r=1, s'=s). Fixed for future generation
  (`final_observation`, `final_state`); existing files keep the limitation (validator warning `terminal_next_obs_copy`).

### A-23 LOW: distractor clearance is not guaranteed
- `ParametricFrankaEnv._sample_distractor_xy` returns its last sample after 50 failed tries, and bodies move while
  settling: in 7 of 76 distractor tasks a distractor ends closer than 9 cm to the start or goal (minimum 3.9 cm,
  task 29). The paper states "at least 9 cm".

### A-24 MEDIUM: competitor and citation errors (from `LITERATURE_COMPARISON.md`, spot-checked before use)
- main.tex credits LIBERO-X with a ">90% to near 0%" collapse (LIBERO-X reports 29 to 65% at L1 and 4 to 18% at L5);
  says Factory transfers to a real Franka (only IndustReal does); misses LIBERO-Plus (7 factors, its own L1-L5,
  10,030 tasks), the closest prior work. About ten bib entries have wrong titles, authors, or arXiv ids.

### A-25 CRITICAL (external exposure): fabricated results are publicly served
- `gh api repos/s-elim/AFFORD_Bench/pages` (2026-10-17): the repository is private, but GitHub Pages is built from
  `main` at the repository root and is public (`https://s-elim.github.io/AFFORD_Bench/`). The committed
  `docs/index.html` shows the synthetic "VLA & Policy Leaderboard" (A-02) and the unsupported comparison figure
  (A-03), and the Pages URL links the author's GitHub account to the paper (a double-blind risk for RSS).
- Local fix done 2026-10-17 (page corrected, synthetic artifacts quarantined). The public site changes only after a
  commit and push, or if Pages is disabled; both need the author's approval (D6, D8).

### A-26 HIGH: replayed demonstrations do not always reproduce their success step
- E-007: 35 of 630 replayed demos (5.6%) reach success at a different step (or not at all by T) although the state
  trajectory deviates by a median 0.02 mm. Labels near the at-rest threshold are numerically sensitive, and the
  data cannot be bit-reproduced from `states[0]` + actions. Not caused by the solver warm start (E-007 diagnostics).
- Root cause (E-012, 2026-10-18): `BaseOracle.act` executed the float64 action but recorded `float32(action)`, so
  every stored action is a rounded copy of what was run. A full diff of `MjData` before the first action shows no
  other difference. Fixed: the oracle now executes the float32 action it records; fresh episodes replay
  bit-identically (`tests/test_replay.py`). Existing files keep the limitation until regenerated.

### A-29 HIGH: perturbation suites were not paired by seed (fixed 2026-10-18)
- The light, color, friction, and mass draws consumed the same random stream before the start pose was sampled, so
  a seed gave starts 5 to 14 cm apart in nominal, p1, p3, and p4 (measured on task 61). The paper's "at identical
  visual coordinates" and the evaluator's "paired seeds" were false per episode; suite comparisons mixed the
  perturbation with a different layout. Fixed: model perturbations draw from a separate stream
  (`engine.MODEL_STREAM`); nominal and p2 episodes are unchanged (a released demo's initial state is still reproduced
  bit-exactly), and `tests/test_perturbation_isolation.py` checks pairing and that each suite changes only its own
  parameters. L4 numbers measured before the fix (E-004, E-008) are being re-measured (E-013).

### A-27 HIGH: the in-repo diffusion baseline does not learn
- E-011: with the same data, budget, and evaluation, low-dim BC solves 16/20 and the in-repo DDPM "diffusion
  policy" 0/20. Reporting it as "Diffusion Policy" would understate that method. Use a reference implementation.

### A-28 MEDIUM: the force threshold no longer separates the expert
- E-009: the oracle exceeds F_max = 150 N in 9/210 episodes (max 163 N); the paper's rationale ("flags contact
  harder than the scripted expert's in any task") no longer holds. Paper now reports the oracle's FVR (4.3%).

## Confirmed as correct (checked, no action)
- 210 core task definitions, all registered; 210 nominal datasets on disk, 50 demos each, 10,500 total (E-002).
- Image shapes 640x660x3 and 300x420x3 uint8; masks 45 train / 5 valid per file; `states` (T, 85) float64;
  `init_state` attribute equals `states[0]` (4 files fully checked in E-002, all files in the validator run).
- Terminal semantics in the 4 checked files: rewards and dones are 0 before the last step and 1 at the last step.
- Each demo has its own seed and a distinct initial state (4 files; all files in the validator run).
- Within each task, generation seeds (`42 + 1000n + k`, k < 400) and evaluation seeds (`200000 + 1000n + j`) are
  disjoint, also for the 8 files from other seed schemes.
- No grasp weld, release snap, or goal teleport in `ParametricFrankaEnv`; success requires release, support, rest,
  tolerance, tilt, and a minimum lift (`parametric_env.check_success`).
- L1 to L4 parameters in `PerturbationConfig` match the paper; nominal model parameters are restored and the sampler
  is re-seeded at every reset (`PerturbationEngine.begin_episode`).
- Control: 25 substeps of 2 ms (20 Hz), action scale 0.025 m, 0.5 m/s cap, yaw 0.05 rad/step, pads with solref 4 ms,
  elliptic cones with impratio 10 (current XML), gripper about 35 N per finger.
- 1722 unit tests pass (E-001).

## Hypotheses and items that need experiments
- H-A: recorded actions replayed from `states[0]` with the current code reproduce each demo's states and its success
  at step T (determinism of the released data under the current physics). Test: validator replay mode (E-005).
- H-B: the L4 slip statements in Sec. IV (elliptic cones hold centered shaft and mug grasps with < 5 mm slip) have no
  artifact; `docs/reviews/l4_slip_probe_2026-10-08.csv` predates the elliptic-cone change. Re-run `scripts/l4_slip_probe.py`.
- H-C: each perturbation mode changes only its intended factor (no change in reachability or start distribution for
  L3/L4). Test: perturbation-isolation tests.
- H-D: dataset start states are conditioned on oracle success (5.4% of sampled starts were discarded). Whether the
  discarded starts cluster (e.g. far scatter) is unknown; it biases training rho0 relative to evaluation rho0.
- H-E: the 35 reach tasks' target points lie on the object the instruction names (e.g. task 82 "touch the top
  surface of the red block"). Test: runtime task audit.
- H-F: Domain 8 (frontier) tasks: README says "14 validated"; no current artifact supports it.
