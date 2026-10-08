# AFFORD-Bench dataset audit against LIBERO, LIBERO-PRO, and LIBERO-X

Date: 2026-10-08. Code at `main` f37e3c5. MuJoCo 3.2.7, Python 3.13, Panda meshes from MuJoCo Menagerie in place of the
absolute `D:/` asset paths. Literature facts were read from primary sources: arXiv HTML via the Hugging Face papers
mirror, and the LIBERO repo at commit 8f1084e and OpenVLA repo at c8f03f4, cloned and read directly.

## Verdict

No, not yet, and the three are different kinds of artifact.
- **LIBERO** is a released dataset of human teleoperation demos with a fixed evaluation protocol.
- **LIBERO-PRO** is an evaluation-only perturbation protocol on top of LIBERO.
- **LIBERO-X** is a new teleoperated training set plus a five-level evaluation.
- **AFFORD-Bench** today is a task suite and a scripted-demo generator. No dataset has been released.

As a benchmark specification, AFFORD-Bench is stricter than LIBERO in two areas:
- Success is checked physically (released, at rest, within tolerance), not by a region predicate.
- It perturbs physics (friction, mass), which none of the three do.

As a dataset it falls below LIBERO's release quality in several ways:
- The recorded reward and done signals are lost.
- No simulator states or fixed initial-state files are stored.
- The "robomimic HDF5" format fails robomimic's own loader.
- The README claims depth that the batch generator does not record.
- 98 of 210 instructions, including every "long-horizon" and "chain" task, are not verified by the success predicate.

Fixes 1–5 below should land before any release or any claim of LIBERO-comparable data quality.

## Side-by-side

| Axis | LIBERO | LIBERO-PRO | LIBERO-X | AFFORD-Bench (measured here) |
| --- | --- | --- | --- | --- |
| Source | arXiv 2306.03310 | arXiv 2510.03827 | arXiv 2602.06556 | this repo |
| Training data | 50 demos/task, human SpaceMouse teleop, 130 tasks (6,500) | none (evaluation only) | 2,520 demos, 600 tasks, 100 scenes, VR teleop; paused or non-optimal trajectories discarded | generator only; nothing released. README's 10,500 is a projection |
| Demo policy | human, multimodal | n/a | human | privileged scripted oracle; episodes deterministic per seed |
| Control | OSC_POSE, 20 Hz, 7-D | same | Cartesian delta, 20 Hz | joint servos via IK, 20 Hz, 7-D; dims 3–4 (roll, pitch) never executed and always 0 |
| Images | agentview and eye-in-hand, 128×128; OpenVLA flips them 180° and re-renders at 256×256 | as LIBERO | rendered from action replay | 640×660 and 300×420, upright (checked) |
| Depth | off by default | n/a | not stated | off in the batch generator (`generate_all_150_tasks.py`); only `generate_demos.py --depth` records it |
| Sim states / replay | states stored; OpenVLA replays and drops failures | init files | replay-rendered | no states stored; open-loop replay reproduces 62/63 successes with state drift up to 1.3e-2 |
| Evaluation inits | 50 fixed init states per task | perturbed inits (position x0.1–0.5) | L1–L2 jitter and spatial randomization | reset seeds (disjoint from demo seeds by default); no exported init-state files |
| Success | BDDL predicate conjunction; `In` fires on boundary breach (LIBERO-X App. D) | as LIBERO | adds ExactIn, UprightOn, SideOn | released, supported, at rest, within 2–8 mm tolerance tiers and tilt bound, lifted ≥ 2 cm |
| Multi-step tasks | LIBERO-10 goals are predicate conjunctions | as LIBERO | subtask-count tags | 98/210 instructions not covered by success; D5–D7 check one placement (`configs/prompts_augmented.yaml`) |
| Language | 1 per task | 3 paraphrases per task | L5: 5 paraphrase types | 1 stored per demo; 4 variants per task exist in config but are not written |
| Perturbations | init states only | object, position, semantic, task, environment | L1–L5: layout, topology, appearance and unseen objects, language | L1 scatter ±7 cm and ±45°, L2 spatial, L3 visual, P1 object, **L4 friction and mass**; fixtures and goals fixed per task |
| Reported headroom | saturated: OpenVLA-OFT 97.1%; LIBERO-PRO reports ~98% for OpenVLA, pi0, pi0.5 | >90% collapses to 0.0% | pi0.5 65.2 → 18.0 (L1 → L5) | oracle 99.8% nominal, 47.3% L4; no learned-policy results in this audit |

## Findings, ranked by severity

**1. Reward and done signals are dropped (soundness, blocks offline RL).**
- Cause: `BaseOracle.generate_episode` seeds `rewards` and `dones` with a reset entry, then slices `[:T]`, which
  drops the final step.
- Effect: every demo stores all-zero rewards and dones. Measured on 5/5 rendered demos and on a bayonet episode that
  succeeded.
- The README advertises CQL and IQL, which need these signals.
- Fix: store the post-action values `rewards[1:T+1]` and `dones[1:T+1]`, and add a test that `dones[-1] == 1` for
  every successful demo.

**2. No released dataset.** `datasets/` is empty and neither the README nor the project page links to hosting. The
"10,500 demos" badge describes what the generator would produce. The demos have also not been regenerated with the
hardened controller (`reality_check_hardening_2026-10-08.md`). LIBERO and LIBERO-X ship data; this is the largest
gap in a "same as LIBERO" claim.

**3. Not replayable, not re-renderable.**
- No `states` dataset, model XML, or `env_args` is stored.
- Replaying the stored actions from the same seed reproduced 62 of 63 successes across 21 tasks.
  `kitchen_tomato_to_plate` failed once, and final joint states drifted by up to 1.3e-2.
- Without states, the data cannot be re-rendered at 128, 224, or 256 px, or filtered by replay as OpenVLA did for
  LIBERO.
- Fix: per step, store the full physics state (`mj_getState`, mjSTATE_PHYSICS) and the compiled model XML. Also
  export fixed initial states per task (for example 50, as LIBERO does), so evaluation does not depend on the RNG
  path through the reset code.

**4. "Robomimic HDF5" is not loadable by robomimic.** robomimic 0.2.0 `scripts/train.py` calls
`get_env_metadata_from_dataset`, which reads `f["data"].attrs["env_args"]` (`utils/file_utils.py` line 82; checked
in the 0.2.0 sdist). AFFORD files do not have this attribute. The obs keys (`agentic_rgb`, `wrist_rgb`, `tcp_pose`)
also differ from the robomimic and LIBERO names (`agentview_rgb`, `eye_in_hand_rgb`, `ee_states`, `joint_states`)
that `verify_dataset_integrity.py`'s own docstring lists.

**5. Instruction–success mismatch on 98/210 tasks.**
- `prompts_augmented.yaml` marks `instruction_covered_by_success: false` for all of D5, D6, D7 and 8 D1 tasks.
- Example: `long_horizon_transmission_assembly` asks for shaft, gear, and pin, but success checks only the shaft.
- All 420 measured oracle episodes contain exactly one grasp and one release.
- LIBERO-10 goals are predicate conjunctions. Until multi-body sequential predicates exist (F6 in the RSS audit),
  D5–D7 should not be called long-horizon or compared to LIBERO-10.

**6. Depth claim not met by the documented pipeline.** README §1 and §3 state 3D metric depth for the 10,500 demos.
`generate_all_150_tasks.py` never passes `render_depth=True`, so the recommended command records no depth.

**7. Storage and resolution.**
- Measured: 139 MiB for 5 demos (597 steps), or 0.23 MiB per step with gzip.
- Projection: at the measured mean of 113 steps per episode, 10,500 demos come to roughly 270 GiB. This is my
  extrapolation from one industrial task; image content changes the compression ratio.
- Each workcell frame has 25.8× the pixels of a LIBERO frame, and the task region fills only part of it (inspected).
  Most VLAs train at 224–256 px.
- Recommendation: release a 256×256 version (or 224) alongside states, as OpenVLA did for LIBERO, and keep
  full-resolution rendering as a re-render option.

**8. Single instruction per demo.** Four prompt variants per task exist in config but are not written to the HDF5.
LIBERO-PRO and LIBERO-X both test paraphrase robustness, so the variants should go into the data (training split) or
into a declared evaluation split.

**9. Demonstration distribution.** The demos are scripted, closed-loop, and privileged; they are deterministic per
seed and unimodal. Episode lengths within a task are nearly constant (118, 118, 118, 118, 125 in the rendered
sample).
- This is clean data: across 420 episodes on all 210 tasks, no-op steps are 6.7% (median 6.9% per episode, max
  17.2%), idle tails are at most 5 steps, there is no action saturation, and there are exactly 2 gripper switches per
  episode.
- It is not human-like: there are no pauses, retries, or path diversity of the kind LIBERO and LIBERO-X teleop
  contain or filter out.
- Results trained on it are therefore not directly comparable to LIBERO-trained results. A policy may also learn the
  oracle's re-centering and settling behaviors.

**10. Memorization surface.** Only the manipulated body scatters (L1 ±7 cm, ±45°), plus ±1.5 cm of TCP jitter.
Fixtures and goal receptacles sit at fixed positions per task. LIBERO-PRO found policies keying on exactly such
layouts. There is no counterpart of LIBERO-X L3 (scene topology) or of LIBERO-PRO's object swap with fixed task.

**11. Minor issues.**
- **Seed overlap:** evaluation seeds are disjoint from demo seeds at the defaults, but `evaluate_id.py` with
  `--num-episodes` > 42 overlaps task 10's demo seeds (10042 onward).
- **Validation split:** the 90/10 split is by demo index.
- **Assets:** everything depends on substituted Menagerie meshes; rerun on the author's assets before quoting
  numbers.

## Strengths relative to the LIBERO family

- **Success is verified physically.** There is no region-entry false positive of the kind LIBERO-X documents for
  `In`. Null and random policies score 0, and no task is solved at reset.
- **L4 physical perturbation** (friction, mass) and contact-only articulated mechanisms (Family C) are absent from
  LIBERO, LIBERO-PRO, and LIBERO-X.
- **Images are upright and well-formed** (inspected); there is no flip convention to undo.
- **Generation is deterministic** and stays clean of failures: only successful episodes are kept.
- **Official LeRobot and RLDS converters** exist.

## README statements to correct

- **LIBERO-PRO:** the paper names 4 perturbation axes and the README lists 5. "20 evaluation suites" is AFFORD's
  arithmetic, not a LIBERO-PRO count.
- **LIBERO-X:** it is not only "domestic tabletop". It has Kitchen and Study scenes and **ships 2,520 new training
  demos**.
- **Scale:** "5.25× scale" compares 210 tasks to LIBERO-PRO's 40. LIBERO-X has 600 tasks and LIBERO-Plus (arXiv
  2510.13626) has 10,030, so the scale claim does not hold against the family.
- **Data claims:** "10,500 demos", "3D metric depth", and "Robomimic HDF5" describe intended output and fail on the
  current code (findings 2, 4, 6).

## Fixes, ranked by impact over effort

| # | Fix | Effort |
| --- | --- | --- |
| 1 | Reward and done off-by-one, plus a test | 10 lines |
| 2 | Write `env_args`, per-step physics `states`, model XML, all prompt variants; robomimic key aliases | ~60 lines |
| 3 | `render_depth` flag in the batch generator, or drop the depth claim | 2 lines |
| 4 | Export fixed init states (50 per task) and evaluate from them | ~80 lines |
| 5 | Correct the README comparisons and the D5–D7 "long-horizon" wording | text |
| 6 | 256×256 release with a replay-based re-render and filter pass; host it with a dataset card | compute and hosting |
| 7 | Multi-body sequential predicates for D5–D7 (F6) | large |
| 8 | Optional human or noise-injected demos for multimodality; fixture and goal layout perturbation | large |

## Not verified

- The exact MuJoCo version LIBERO used, and the count of failed demos in raw LIBERO.
- LIBERO-X's test-set size and controller details.
- How AFFORD data behaves when actually trained in robomimic or LeRobot (no training run here).
- The rendered sample covers one task (5 demos). Action statistics cover all 210 tasks at 2 seeds each, but without
  images.
