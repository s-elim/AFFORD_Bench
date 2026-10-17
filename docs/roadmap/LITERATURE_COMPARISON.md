# Literature and Benchmark-Comparison Audit (AFFORD-Bench, RSS 2027)

Audit date: 2026-10-17. Scope: the related-work and comparison claims in `README.md`,
`AFFORD-Bench_RSS'27/paper-template-latex/main.tex`, `references.bib`, and the figure
`figures/benchmark_comparison.png` (made by `scripts/generate_benchmark_comparison_figure.py`).

Verdicts:
- **SUPPORTED**: a primary source (paper, official repo, official dataset card, or code) states it.
- **CONTRADICTED**: a primary source states something else. The correction is given.
- **UNVERIFIED**: I could not confirm it from a primary source in this audit. Do not state it as fact.

Evidence order: local paper text (`scratch/libero_pro_raw.txt` = arXiv 2510.03827v2; `scratch/libero_x_raw.txt` = arXiv 2602.06556v1),
local code (`../LIBERO-PRO`, `../LIBERO-X`, `../robomimic`), then arXiv abstract/API, official GitHub READMEs,
Hugging Face dataset cards, Crossref DOI records, PMLR/RSS proceedings pages. Numbers marked "(derived)" are my arithmetic on sourced numbers.

Note: the comparison figure is **not** included in `main.tex` (its only `\includegraphics` is `tasks_horizontal_vis.pdf`). It is shown in `docs/index.html`. The corrections in Section 3 still apply if it goes into the paper.

---

## 1. Comparison table

### 1a. Core comparators

| Benchmark | # tasks | Demos shipped (count, source) | Cameras / resolution | Perturbation axes | Friction / mass perturbed | Tolerance indexing | Success definition | Policies evaluated | Licence | Primary source | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LIBERO** | 130 in 4 suites: Spatial 10, Object 10, Goal 10, LIBERO-100 (split 90 + 10) [paper §4.2] | 50 human SpaceMouse demos per task [paper §4], so 6,500 (derived) | agentview + eye-in-hand RGB, 128×128 in LIBERO's default data config and env wrapper (`libero/configs/data/default.yaml`, `envs/env_wrapper.py`). Not stated in the main paper text | None as a benchmark axis (small initial-state jitter only) | N | N | Conjunction of BDDL goal predicates. Object `On` = contact + z order + xy centre distance < 3 cm (`object_states/base_object_states.py` l.78-93). No release or at-rest condition | Lifelong BC (ResNet-RNN, ResNet-T, ViT-T) in the paper. De facto VLA testbed later | Code MIT, datasets CC BY 4.0 (GitHub README) | arXiv 2306.03310; NeurIPS 2023 D&B | High |
| **LIBERO-PRO** | 40 base tasks (4 LIBERO suites × 10). The HF card adds a 7-case extension of 280 BDDL files (7 × 40) | **No new demos.** Evaluates checkpoints trained on LIBERO's demos (paper §5.1; HF dataset holds only `bddl` + `init` files) | Inherits the LIBERO env (two cameras). Resolution is set by each model's eval script and is not stated in the paper | Paper abstract: 4 (manipulated objects, initial states, task instructions, environments). Repo/HF: 5 (Object, Position, Semantic, Task, Environment). HF extension: glare/noise, camera view, runtime object move, texture, occlusion, shape, initial pose | N (no friction/mass code in `perturbation.py` or `libero_ood/`) | N | LIBERO predicates, 50 episodes per task | VLAs only. Paper: OpenVLA, π0, π0.5. Repo leaderboard adds UniVLA, MolmoAct, NORA, X-VLA | Code MIT (LICENSE keeps "Copyright (c) 2023 Lifelong Robot Learning"), data CC-BY-4.0 (HF card) | arXiv 2510.03827 (v1 4 Oct 2025, v2 25 May 2026); no venue found | High |
| **LIBERO-Plus** (real, and missing from the paper) | 10,030 tasks over 7 factors and 21 components, stratified into **L1-L5** difficulty levels [paper §6.1] | ">20,000 successful trajectories" for training [paper §6.2]. The repo links RLDS and LeRobot releases. Released count UNVERIFIED | Camera viewpoint is a perturbation axis. Resolution UNVERIFIED | Objects layout, camera viewpoints, robot initial states, language, light, background textures, sensor noise | N (abstract and repo list no physics axis) | N | Built on LIBERO. Exact predicate changes UNVERIFIED | 10 VLAs (OpenVLA, OpenVLA-OFT ×3 variants, π0, π0-fast, NORA, WorldVLA, UniVLA, RIPT-VLA) | UNVERIFIED | arXiv 2510.13626 (v1 15 Oct 2025, v3 26 Dec 2025). CVPR 2026 poster "LIBERO-Plus: A Progressive Robustness Benchmark for VLA Models", same 13 authors | High (content), Medium (venue title) |
| **LIBERO-X** | 600 tasks in 100 scenes built on **Kitchen and Study** base scenes [paper §III-A-2]. Local L1 BDDL: 287 KITCHEN + 313 STUDY | 2,520 human VR-teleop demos (Meta Quest 3, via LeRobot) [paper; README; HF `meta/info.json`: 2,520 episodes, 889,277 frames, fps 10, `total_tasks` 428] | 2 RGB (third-person + wrist), 256×256 (HF `info.json`). Eval renders at 256 and resizes to 224 (`eval_template.py`) | Cumulative L1 local spatial, L2 extended spatial, L3 scene topology, L4 visual attributes/unseen objects, L5 semantic reformulation | N (only arena table friction in code) | N | LIBERO-style BDDL predicates plus `ExactIn` (bottom within 4 cm of table height), `UprightOn` (cos ≥ 0.8, about 36.9°), `SideOn`. No release/rest condition | VLAs fine-tuned on LIBERO-X: OpenVLA-OFT, X-VLA, GR00T-N1.5, π0, π0.5. 10 trials per task | Code MIT ("Copyright (c) 2026 Meituan"), data CC BY 4.0 | arXiv 2602.06556; **RSS 2026** (RSS XXII proceedings, p097) | High |
| **THE COLOSSEUM** | 20 RLBench tasks | Sim: 100 scripted motion-planner demos per task. Real: 5 per task (HTC Vive) [paper] | Multi-view RGB-D for the baselines. Resolution in sim UNVERIFIED (real Kinect-2 is 512×424) | 14: MO_Color, MO_Texture, MO_Size, RO_Color, RO_Texture, RO_Size, Light_Color, Table_Color, Table_Texture, Distractor, Background_Texture, Camera_Pose, **Object_Friction, Object_Mass** | **Y.** Friction sampled in [0.75, 1.0]. Mass ranges are task-dependent (Table III lists 5 tasks), and the axes apply only to tasks that support them | N | RLBench task-completion conditions | VoxPoser, R3M-MLP, MVP-MLP, PerAct, RVT (not VLAs) | UNVERIFIED | arXiv 2402.08191; RSS 2024 (DOI 10.15607/RSS.2024.XX.133) | High (factors), Medium (physics task coverage) |
| **Factory** | Simulation methods + 60 part models, 3 assembly environments, 7 controllers [abstract] | None (RL) | n/a | Not a perturbation benchmark | n/a | Assets have defined clearances. Not a swept axis (UNVERIFIED detail) | Task-specific | Proof-of-concept RL for nut-and-bolt, **in simulation only** [abstract] | UNVERIFIED | arXiv 2205.03532; RSS 2022 | High |
| **IndustReal** | Contact-rich pick, place, insertion | None (RL) | n/a | n/a | n/a | Not a swept axis (UNVERIFIED) | Task-specific | RL, sim-to-real on a real robot [abstract] | UNVERIFIED | arXiv 2305.17110; RSS 2023 | High |
| *AFFORD-Bench (for reference)* | 210 tasks, 7 domains | 50 privileged-oracle demos per task, 10,500 planned (main.tex l.239) | 640×660 workcell + 300×420 wrist RGB, optional depth | L1 nominal scatter, L2 spatial, L3 light/colour, L4 friction [0.12, 0.25] + mass ×[1.8, 2.5], L5 compound | Y, on all tasks (`afford_pro/perturbations/engine.py` l.42-43) | **Fixed per-task tiers (2-2.5 mm / 8 mm). Tolerance-indexed variants are a Stage 2 deliverable** (main.tex Table caption l.185, H2 l.257-260) | Released, supported, at rest, within tolerance, after a ≥2 cm lift | Oracle, random, null only. BC/DP pending | **No LICENSE file in repo** | this repo | n/a |

### 1b. Other benchmarks a reviewer may raise (each exists; identifiers checked on arXiv)

| Benchmark | One-line description (from abstract or code) | Physics (friction/mass) perturbation | Tolerance-based success |
|---|---|---|---|
| RLBench (arXiv 1909.12271; RA-L 2020) | 100 hand-designed tasks, motion-planner demos, over-shoulder stereo + wrist cameras | N | N (task conditions) |
| ManiSkill2 (2302.04659; ICLR 2023) / ManiSkill3 (2410.00425) | 20 task families, 2000+ object models, 4M+ demo frames, explicit object-level geometric variation. MS3: GPU-parallel simulation across 12 domains | Not a benchmark axis. Physics randomization support UNVERIFIED | Partly: MS3 `StackCube` success = on-target (xy within half-diagonal + 5 mm, z within 5 mm) **and static and not grasped**. `PegInsertionSide` hole clearance is 3 mm (checked in `mani_skill/envs/tasks/tabletop/*.py`) |
| RoboCasa (2406.02523; RSS 2024) | Kitchen simulation, 100 tasks, human demos plus automated trajectory generation | N (UNVERIFIED beyond abstract) | N |
| ARNOLD (2304.04321; ICCV 2023) | 8 language-conditioned tasks with continuous goal states, Isaac Sim, about 10k demos | N | **Y, fixed**: ±5 cm, ±20°, ±10%, held for 2 s |
| Factor World (2307.03659) | 19 tasks, 11 factors (lighting, object texture/size/shape/position, table/background texture, camera pose, table position, arm pose, distractors) | N | N |
| KitchenShift (NeurIPS 2021 DistShift workshop; OpenReview DdglKo8hBq0) | Kitchen simulation with visual (lighting, camera) and state (object orientation) domain shifts for imitation learning | N (per available summary; full list UNVERIFIED) | N |
| GemBench (2410.01345; ICRA 2025) | 7 action primitives, 4 generalization levels (placements, rigid/articulated objects, long-horizon), on RLBench | N | N |
| VLABench (2412.18194) | 100 task categories, 2000+ objects, language-conditioned long-horizon reasoning | N as a perturbation axis | N |
| AGNOSTOS (2505.15660) | 23 unseen tasks, 2 difficulty levels, cross-task zero-shot VLA generalization | N | N |
| RoboArena (2506.18123) | Crowd-sourced real-world pairwise evaluation (DROID, 7 institutions, 600+ pairwise episodes) | n/a | n/a |
| SIMPLER (2405.05941) | Simulated evaluation correlated with real-robot results | N | N |
| RoboTwin 2.0 (2506.18088) | 50 dual-arm tasks, 5 embodiments. Domain randomization over clutter, lighting, background, table height, language | N | N |
| FurnitureBench (2305.12821; RSS 2023) | Real-world furniture assembly, 5000+ demos, FurnitureSim | UNVERIFIED | UNVERIFIED |
| AutoMate (2407.08028) | 100 assemblies, specialist/generalist assembly policies, zero-shot sim-to-real | UNVERIFIED | Assembly-specific (UNVERIFIED) |
| **ManipulationNet** (2603.04363, Mar 2026) | Real-world distributed benchmark. Peg-in-hole with 5 peg shapes × **4 clearance levels (3, 1, 0.1, 0.02 mm)** | n/a (real world) | **Y, real-world clearance-indexed** |

I found no simulated benchmark that sweeps success tolerance as an independent axis together with friction/mass shifts for imitation or VLA policies. This rests on a targeted search, not an exhaustive one, so treat it as UNVERIFIED.

---

## 2. Claim-by-claim verdicts

### Item 1: LIBERO (arXiv 2306.03310)

| # | Claim (where) | Verdict | Source / correction |
|---|---|---|---|
| 1.1 | 130 tasks, 4 suites (README l.22, l.33; main.tex l.144) | SUPPORTED | arXiv abstract; paper §4.2 |
| 1.2 | "130 tasks, **50 core**" (README l.22, l.33) | **CONTRADICTED** | LIBERO has no "core" subset. Suites are Spatial/Object/Goal (10 tasks each) and LIBERO-100 (split into LIBERO-90 and LIBERO-10). Write "130 tasks in four suites (10/10/10/100)". If you mean the usual VLA protocol, it is the **40** tasks of Spatial, Object, Goal and LIBERO-10 |
| 1.3 | 6,500 human SpaceMouse teleop demos, 50 per task (README l.33) | SUPPORTED | Paper §4 (just before §5): "50 trajectories of high-quality demonstrations for every single task ... teleoperation with 3Dconnexion Spacemouse". 130 × 50 = 6,500 (derived) |
| 1.4 | Success by BDDL predicate conjunctions (README l.33) | SUPPORTED | Paper §4 (goal = "a conjunction of predicates"; "terminates when all predicates are verified true"). Code `_check_success`. Note: no release or at-rest requirement, and `On` accepts up to 3 cm xy centre offset |
| 1.5 | Cameras agentview + eye-in-hand at 128×128 | SUPPORTED (code), not stated in the main paper | `libero/configs/data/default.yaml` (`img_h/img_w: 128`, `agentview_rgb`, `eye_in_hand_rgb`). Many VLA eval scripts re-render at 256 |
| 1.6 | Physics-based grasping in robosuite, not welds | SUPPORTED (medium-high) | Paper: built on robosuite. No `weld` anywhere in the LIBERO code copies in LIBERO-PRO/LIBERO-X. The robosuite gripper source was not inspected |
| 1.7 | Licence | SUPPORTED | Code MIT, datasets CC BY 4.0 (github.com/Lifelong-Robot-Learning/LIBERO) |

### Item 2: LIBERO-PRO (arXiv 2510.03827)

| # | Claim | Verdict | Source / correction |
|---|---|---|---|
| 2.1 | Evaluation-only | SUPPORTED | HF `zhouxueyang/LIBERO-Pro` holds only `bddl_files`, `init_files`, `metadata`. Paper §5.1 uses official checkpoints |
| 2.2 | 40 tasks (README l.22, l.34; figure) | SUPPORTED as **40 base tasks** | 4 suites × 10 (v1 tables; HF card). The HF 7-case extension has 280 BDDL files |
| 2.3 | 4 perturbation axes "(object, spatial, semantic, environment)" (README l.34) | **PARTLY CONTRADICTED** | Abstract: four dimensions, "manipulated objects, initial states, task instructions, and environments". The repo and HF list **five** (Object, Position, Semantic, **Task**, Environment), and the HF card adds lighting/noise, camera view, occlusion, runtime object motion, shape, initial pose. Suggested: "four perturbation dimensions (objects, initial positions, instructions incl. task changes, environments), extended in the released code to five plus seven robustness cases" |
| 2.4 | Models evaluated | SUPPORTED: VLAs only | Paper: OpenVLA, π0, π0.5. Repo leaderboard adds UniVLA, MolmoAct, NORA, X-VLA |
| 2.5 | Ships demos? | No | Training data is LIBERO's |
| 2.6 | Licence | MIT code, CC-BY-4.0 data | `LICENSE` (MIT, copyright line inherited from LIBERO); HF card |
| 2.7 | Posted Oct 2025 | SUPPORTED | v1 4 Oct 2025; v2 25 May 2026 (arXiv abs page). No peer-reviewed venue found (UNVERIFIED that none exists) |
| 2.8 | main.tex l.147: policies "collapse under perturbed objects, initial states, and instructions" | **PARTLY CONTRADICTED** | Repo leaderboard: OpenVLA keeps 0.81-0.98 under Object and 0.96-0.98 under Semantic (paraphrase). **0.00 under Position and Task** in all four suites. Suggested: "collapse under changed initial positions and task changes, while apparent robustness to object-appearance and paraphrase changes reflects insensitivity to those inputs" |
| 2.9 | main.tex l.59 ">90% nominal → near 0% under small spatial offsets, object swaps, or altered instructions", attributed to LIBERO-PRO **and LIBERO-X** | SUPPORTED for LIBERO-PRO (position/task); **CONTRADICTED for LIBERO-X** | See 3.8 |

### Item 3: LIBERO-X (arXiv 2602.06556)

| # | Claim | Verdict | Source / correction |
|---|---|---|---|
| 3.1 | 600 tasks | SUPPORTED | Paper §III-A; README; local `bddl/LEVEL1..3` have 600 files each (LEVEL4 has 826). Caveat: HF `meta/info.json` says `total_tasks: 428`, probably unique instruction strings (UNVERIFIED) |
| 3.2 | 2,520 VR teleop demos | SUPPORTED | Paper: VR teleop with Meta Quest 3, via LeRobot. HF: 2,520 episodes |
| 3.3 | 5-level hierarchy L1-L5 | SUPPORTED | Cumulative levels (paper §III-B) |
| 3.4 | "Kitchen, Study, and **Living** scenes" (README l.22) | **CONTRADICTED** | "built upon Kitchen and Study base scenes" (paper). Local L1 BDDL: only KITCHEN and STUDY prefixes, 100 distinct scenes |
| 3.5 | 2× RGB 256×256 | SUPPORTED | HF `info.json`: `image` and `wrist_image` 256×256×3 |
| 3.6 | Licence | SUPPORTED | MIT code, CC BY 4.0 data (README table; HF card) |
| 3.7 | Authors / Meituan | SUPPORTED | Wang, Zhang, Liu, Zhang, Cai, Liu, Liu. Affiliations Meituan and Beihang University. LICENSE copyright Meituan |
| 3.8 | Shows ">90% nominal → near 0%" (main.tex l.59) | **CONTRADICTED** | LIBERO-X Table II: L1 success 29.0-65.2%, L5 4.1-18.0% (the paper reports an "average performance decline of 31.2%" from L1 to L5). Its finding is that models trained on LIBERO-X fail even at L1, not a collapse from >90% |
| 3.9 | Venue | Update needed | Accepted to RSS 2026 (README, HF card). Listed in RSS XXII proceedings as p097 |

### Item 4: Figure cells for LIBERO-PRO / LIBERO-X (`generate_benchmark_comparison_figure.py`)

| Cell | Verdict | Source / correction |
|---|---|---|
| "Nominal (Fixed μ = 0.5)", both | "No friction/mass perturbation" is SUPPORTED. **"μ = 0.5" is CONTRADICTED** | LIBERO object geoms use `friction="0.95 0.3 0.1"` (1,618 occurrences in LIBERO-PRO assets, 1,908 in LIBERO-X). Tables use 1.0 (arena default) or 0.6 (`bddl_base_domain.py` l.312). No 0.5 anywhere. Gripper-pad values UNVERIFIED. Write "No physics perturbation (fixed nominal friction/mass)" |
| "1× RGB (128 × 128)", LIBERO-PRO | **CONTRADICTED ("1×")**; resolution UNVERIFIED | The LIBERO env provides two RGB cameras (agentview + eye-in-hand). The LIBERO-PRO paper states no resolution, and each model's eval script sets it. Write "2× RGB (LIBERO env; resolution set per model, 128 default)" |
| "2× RGB (256 × 256)", LIBERO-X | SUPPORTED | HF `info.json`; `eval_template.py` |
| "VLA Models Only", both | **Misleading.** Accurate only as "only VLAs were evaluated" | Both repos ship LIBERO's BC-RNN, BC-Transformer and BC-ViLT policies (`libero/lifelong/models/`). LIBERO-X data is LeRobot format, usable by any IL policy, and the paper converts it to RLDS for OpenVLA-OFT. Write "Evaluated: VLAs only". Your "Universal + LeRobot & RLDS" contrast does not hold against LIBERO-X |
| "Coarse / Kinematic Welds", both | **"Kinematic welds" CONTRADICTED. "Coarse" SUPPORTED** | No weld constraints in either codebase; grasping uses the robosuite physical gripper. Coarseness is real: `On` allows 3 cm xy offset, LIBERO-X `ExactIn` 4 cm z, `UprightOn` about 37°, and no release or rest is required. Write "Coarse predicates (cm-scale, no release/rest check)" |
| "Unverified Null Baselines", both | SUPPORTED that **none are reported**. The wording is editorial | Neither paper reports null, random or oracle solvability baselines (LIBERO-PRO v1 HTML; LIBERO-X text search). Fair version: "No null/random or solvability baselines reported". LIBERO-PRO's perturbed tasks have no demonstrations, so their solvability is not shown |
| "40 Eval Tasks (4 Axes)", LIBERO-PRO | Partly | 40 base tasks is correct. Axes are 4 in the paper and 5 (+7 cases) in the code. See 2.3 |
| "600 Procedural Tasks", LIBERO-X | 600 SUPPORTED; "procedural" UNVERIFIED | Paper: "we created 100 scenarios encompassing 600 distinct tasks". No procedural-generation claim found |
| "Eval Only (0 Demos)", LIBERO-PRO | **Misleading** | No new demos, but it is used with LIBERO's 50 human demos per task (2,000 for its 40 tasks, derived). The scale bar shows 19% for "0 demos", which is internally inconsistent |
| "600 Tasks • 2,520 VR Demos", LIBERO-X | SUPPORTED | See 3.1-3.2 |
| Header "Zhou et al., 2026" | **CONTRADICTED** | v1 posted 4 Oct 2025, so "Zhou et al., 2025" |
| Header "Wang et al., 2026 • Language-X" | Year SUPPORTED. **"Language-X" CONTRADICTED** | LIBERO-X targets spatial generalization, object recognition and instruction understanding (abstract). Write "Robustness litmus" or "Hierarchical L1-L5" |
| AFFORD cell "Universal + LeRobot v2.0 & RLDS" | Not a differentiator | LIBERO-X ships LeRobot data; LIBERO-Plus links RLDS and LeRobot data |
| "Benchmark Scale: Normalized data volume" bars | UNSUPPORTED visual | The bars are hand-set (0.19, 0.55, 1.0). They compare a 10,500-demo *target* to released datasets and ignore LIBERO (6,500) and LIBERO-Plus (>20,000) |

### Item 5: THE COLOSSEUM (Pumacay et al., RSS 2024)

| # | Claim (main.tex l.120, l.147) | Verdict | Source |
|---|---|---|---|
| 5.1 | 20 RLBench tasks | SUPPORTED | Abstract; repo README ("20 out of the original 100 tasks from RLBench") |
| 5.2 | 14 perturbation factors | SUPPORTED | Abstract ("14 axes of environmental perturbations") |
| 5.3 | Physical properties (friction, mass) included | SUPPORTED, with a caveat to add | Object_Friction ([0.75, 1.0]) and Object_Mass (task-dependent ranges, Table III lists 5 tasks). They apply only to tasks that support them, and the paper calls physical-perturbation results for most models "inconclusive" |
| 5.4 | "30-50% for single factors, ≥75% for combined" | SUPPORTED | Abstract: drops "between 30-50%" across factors and ≥75% under combined perturbations |

### Item 7: `references.bib`

| Key | Verdict | Correction |
|---|---|---|
| liu2023libero | OK | NeurIPS 2023 Datasets & Benchmarks. 7 authors match |
| zhou2026liberopro | OK (content) | Title, 8 authors, arXiv id, year 2025 all correct. The key says 2026 (cosmetic, but the figure copied the wrong year) |
| wang2026liberox | OK, **update venue** | `@inproceedings{..., booktitle={Robotics: Science and Systems (RSS)}, year={2026}}` (RSS XXII, p097) |
| mandlekar2021robomimic | OK | CoRL 2021, PMLR 164, pp. 1678-1690 (PMLR year 2022) |
| chi2023diffusion | **WRONG authors** | Cheng Chi, Siyuan Feng, **Yilun Du, Zhenjia Xu**, Eric Cousineau, Benjamin Burchfiel, Shuran Song (RSS 2023, DOI 10.15607/RSS.2023.XIX.026). "Pan, Yilun", "Feng, Zhenjia" and "Xu, Danfei" are wrong |
| zhao2023act | OK | RSS 2023 (DOI 10.15607/RSS.2023.XIX.016) |
| kim2024openvla | **WRONG authors; venue** | Kim, Pertsch, Karamcheti, Xiao, Balakrishna, Nair, Rafailov, Foster, Lam, Sanketi, Vuong, Kollar, Burchfiel, Tedrake, Sadigh, Levine, Liang, Finn. "Siddharth, Siddharth", "Tan, Chelsea" and "Zeng, Andy" are not authors. Venue: CoRL 2024, PMLR 270, pp. 2679-2713 |
| octo2023 | **WRONG authors** | Octo Model Team, Ghosh, Walke, Pertsch, Black, **Mees, Oier**, Dasari, Hejna, Kreiman, Xu, Luo, Tan, Chen, Sanketi, Vuong, Xiao, Sadigh, Finn, Levine. "Zheng, Oier", "Nair, Suraj" and "Lu, Chelsea" are wrong. RSS 2024 is correct (DOI 10.15607/RSS.2024.XX.090) |
| padalkar2023openx | UNVERIFIED details | arXiv 2310.08864 exists. The author list differs from current arXiv v9 ordering, and "Herzog, Alexander" / "Rai, Anikait" look garbled. The ICRA 2024 venue was not confirmed in this audit. Use the project-site BibTeX |
| brohan2022rt1 | OK | RSS 2023 (DOI 10.15607/RSS.2023.XIX.025) |
| brohan2023rt2 | OK (arXiv authors) | CoRL 2023 venue not re-verified here |
| ze2024dp3 | **WRONG title + one author** | "...via **Simple** 3D Representations". "Zhang, Kangrui" should be "Zhang, Kangning". RSS 2024 correct |
| james2020rlbench | **WRONG pages** | RA-L 5(2), **pp. 3019-3026** (Crossref, DOI 10.1109/LRA.2020.2974707) |
| gu2023maniskill2 | OK | ICLR 2023 |
| todorov2012mujoco | OK | IROS 2012, pp. 5026-5033 |
| gibson1979ecological | Not checked | (book) |
| shridhar2022cliport | OK | CoRL 2021, PMLR 164, pp. 894-906 (2022) |
| song2023grasping | **WRONG (authors, venue, year, title)** | Turpin, Wang, Heiden, Chen, Macklin, Tsogkas, Dickinson, Garg. "Grasp'D: Differentiable Contact-rich Grasp Synthesis for Multi-fingered Hands", **ECCV 2022**, arXiv 2208.12250. Also note it targets multi-fingered hands |
| fu2024mobilealoha | OK | arXiv 2401.02117, 3 authors |
| smolvla2025 | OK | arXiv 2506.01844, Shukor et al. (14 authors), 2025 |
| atomvla2025 | **Author order wrong** | Title, arXiv 2603.08519 and year 2026 are correct. Authors: Sun, Xiaoquan; Xu, Zetian; **Cao, Chen**; ... ; Chen, Jiayu (last of 12). The key says 2025. main.tex's description (subtask decomposition + predictive latent world model) matches the abstract |
| pumacay2024colosseum | OK | RSS 2024 |
| narang2022factory | OK | 12 authors and RSS 2022 match |
| tang2023industreal | OK | 8 authors and RSS 2023 match |
| cadene2024lerobot | Incomplete | The official repo BibTeX has 18 authors. The README also gives an ICLR 2026 paper, "LeRobot: An Open-Source Library for End-to-End Robot Learning" (arXiv 2602.22818), which is better to cite |
| kumar2020cql | OK (pages not verified) | NeurIPS 2020, vol. 33 |
| black2024pi0 | **WRONG title + authors** | "π0: A Vision-Language-Action Flow Model for **General Robot Control**". Authors: Black, Brown, Driess, Esmail, Equi, Finn, Fusai, Groom, Hausman, Ichter, Jakubczak, Jones, Ke, Levine, Li-Bell, Mothukuri, Nair, Pertsch, Shi, Tanner, Vuong, Walling, Wang, Zhilinsky. Venue **RSS 2025** (DOI 10.15607/RSS.2025.XXI.010) |
| qu2024spatialvla | **WRONG id, title, authors** | arXiv **2501.15830** (2501.15835 is a 2D-materials physics paper). Title "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model". Authors: Qu, Song, Chen, Yao, Ye, Ding, Wang, Gu, Zhao, Wang, Li. **RSS 2025** |
| zheng2024cotvla | **WRONG id, title, authors** | arXiv **2503.22020** (2410.05260 is "DartControl"). Title "CoT-VLA: Visual Chain-of-Thought Reasoning for **Vision-Language-Action Models**". Authors: **Zhao, Qingqing**; Lu; Kim; Fu; Zhang; Wu; Li; Ma; Han; Finn; Handa; Liu; Xiang; Wetzstein; Lin. **CVPR 2025**. Rename the key |

Missing citations: **LIBERO-Plus** (closest prior work; also uses "L1-L5"), **MuJoCo Menagerie** (the paper uses its Panda meshes, main.tex l.289), ManiSkill3, ARNOLD, RoboCasa, Factor World, GemBench, ManipulationNet, AutoMate.

### Other main.tex statements checked

- l.144 "These suites evaluate mostly nominal execution without systematic OOD testing": **overstated for ManiSkill2**, whose abstract centres object-level geometric variation.
- l.150 "Factory and IndustReal ... transfer RL insertion and gear-meshing policies to a real Franka": **CONTRADICTED for Factory**, which trains proof-of-concept nut-and-bolt RL in simulation only. IndustReal does transfer to the real world. "SDF collisions" is UNVERIFIED from the abstracts (IndustReal's abstract mentions SDF *rewards*).
- l.147 "loose placement tolerances": fair for the LIBERO family (3 cm `On`). Do not generalise to ManiSkill (3 mm peg clearance), Factory/IndustReal/AutoMate, or ManipulationNet (down to 0.02 mm, real world).
- l.57 / l.147 SmolVLA "pairs a compact backbone with flow matching": UNVERIFIED from the abstract (check the paper body).

---

## 3. What AFFORD-Bench would need to show to claim a distinct contribution

1. **Deliver tolerance indexing.** It is the stated differentiator, but main.tex says tolerance-indexed variants and H2 are Stage 2. ARNOLD and ManiSkill already use fixed per-task tolerances, and ManipulationNet sweeps clearance (3 → 0.02 mm) in the real world. The distinct claim needs simulated **success-vs-ε curves for learned policies** (ε ∈ {2, 4, 8, 16} mm at fixed shift), ideally crossed with L4.
2. **Show L4 adds something COLOSSEUM does not.** COLOSSEUM already perturbs friction and mass, but only on a subset of tasks and with inconclusive results. AFFORD should show (a) L4 on all tasks, (b) a **feasibility reference for L4** (an adaptive oracle that solves most L4 episodes; today the oracle gets 47.3% and the paper says solvability is unknown), and (c) learned-policy degradation under L4 that is not predicted by L2/L3 degradation (different rankings or interaction effects).
3. **Scope the release-and-rest predicate as rigor, not novelty.** ManiSkill3 `StackCube` already requires the cube to be static and not grasped, and ARNOLD requires a 2 s hold. Make it quantitative: re-score the same rollouts with a LIBERO-style predicate (contact + 3 cm, no release) and report how much success that over-counts.
4. **Report learned-policy results.** LIBERO-PRO, LIBERO-X and LIBERO-Plus each evaluate 3-10 policies. Without BC/DP/VLA numbers across L1-L5 with confidence intervals, fragility claims are untested. Also show that privileged-oracle demos are learnable at L1: LIBERO and LIBERO-X use human demos, and scripted demos invite questions about multimodality and learnability.
5. **Position against LIBERO-Plus and LIBERO-X explicitly.** Both already use "L1-L5". In LIBERO-Plus they are difficulty strata; in LIBERO-X they are cumulative perturbation levels; AFFORD's levels are perturbation types. Say so, and drop comparison cells that strawman them (μ = 0.5, welds, "VLA only", "0 demos").

---

## 4. Licences and reuse

| Project | Licence (source) | Does AFFORD-Bench reuse it? |
|---|---|---|
| LIBERO | Code MIT; datasets CC BY 4.0 (GitHub README) | No code or data copied (no `import libero` anywhere). Only schema compatibility: HDF5 attribute and key names (`problem_info`, `ee_states`, `tag="libero-v1"`). Cite LIBERO (already done); no licence obligation for format compatibility |
| LIBERO-PRO | MIT (`LICENSE`, copyright line from LIBERO); data CC-BY-4.0 (HF card) | No |
| LIBERO-X | Code MIT (© 2026 Meituan); data CC BY 4.0 | No. Attribution would be needed only if its data is redistributed or converted |
| robomimic | MIT (© 2021 Stanford Vision and Learning Lab, local `LICENSE`) | No code imported. AFFORD writes robomimic-compatible HDF5 (`afford_pro/datasets/hdf5_writer.py`). Cite robomimic (done; the official BibTeX has no pages field) |
| MuJoCo Menagerie `franka_emika_panda` | **Apache-2.0** (folder `LICENSE`). The rest of Menagerie is Apache-2.0, © DeepMind | **Yes.** 67 mesh files in `afford_pro/assets/robots/assets/`. Sampled `hand.stl`, `finger_0.obj` and `link0_0.obj` are identical to Menagerie's upstream files (byte-identical apart from CRLF line endings). The Panda body in all six scene XMLs follows Menagerie `panda.xml` (classes `panda`/`finger`/`visual`/`collision`/`fingertip_pad_collision_1-5`, materials, armature, finger tendon), modified (elliptic cone, impratio, pads). Git history (commit 7ba297d) shows they came via the local `cap-x` project (MIT, © 2026 Max Fu), which does not change the Apache-2.0 terms of these files |
| `afford_pro/assets/textures/cartoon_only.png` | **UNVERIFIED** | Copied from a local `franka_drawing_sim` folder. Origin and licence unknown; settle before release |

Required before public release:
1. Add `afford_pro/assets/robots/LICENSE` with the Apache-2.0 text, a note that the files derive from MuJoCo Menagerie `franka_emika_panda` (itself derived from `franka_ros` `franka_description`), and a statement that the scene XMLs modify `panda.xml` (Apache-2.0 §4(b)). Carry any upstream NOTICE file; I saw none in the folder listing (UNVERIFIED).
2. Cite Menagerie in the paper. Official BibTeX: `@software{menagerie2022github, author={Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}}, title={{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}}, url={http://github.com/google-deepmind/mujoco_menagerie}, year={2022}}`.
3. Add a top-level `LICENSE` for AFFORD-Bench itself (none exists), plus a separate data licence for the released HDF5 demos.

---

## 5. Incidental consistency issues (outside the literature claims)

- README l.36 says 240 tasks, 8 domains, 40 suites, μ ∈ [0.10, 1.20] and mass 0.5-3.0×. The code (`engine.py` l.42-43) and main.tex say 210 tasks, 7 domains, 35 suites, μ ∈ [0.12, 0.25] and mass ×[1.8, 2.5]. The README header badges also say 210/35.
- The figure labels the AFFORD demo count "10,500 Demos Target", while main.tex l.239 describes them as recorded. Pick one.

---

## 6. Sources

- LIBERO: https://arxiv.org/abs/2306.03310 ; NeurIPS PDF https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf ; https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO-PRO: https://arxiv.org/abs/2510.03827 ; https://arxiv.org/html/2510.03827v1 ; https://github.com/Zxy-MLlab/LIBERO-PRO ; https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro ; local `../LIBERO-PRO`, `scratch/libero_pro_raw.txt`
- LIBERO-Plus: https://arxiv.org/abs/2510.13626 ; https://github.com/sylvestf/LIBERO-plus ; https://cvpr.thecvf.com/virtual/2026/poster/38735
- LIBERO-X: https://arxiv.org/abs/2602.06556 ; https://github.com/meituan/LIBERO-X ; https://huggingface.co/datasets/meituan/LIBERO-X ; https://www.roboticsproceedings.org/rss22/index.html ; local `../LIBERO-X`, `scratch/libero_x_raw.txt`
- COLOSSEUM: https://arxiv.org/abs/2402.08191 ; https://arxiv.org/html/2402.08191v2 ; https://github.com/robot-colosseum/robot-colosseum
- Factory https://arxiv.org/abs/2205.03532 ; IndustReal https://arxiv.org/abs/2305.17110 ; AutoMate https://arxiv.org/abs/2407.08028
- ManiSkill2 https://arxiv.org/abs/2302.04659 ; ManiSkill3 https://arxiv.org/abs/2410.00425 and https://github.com/haosulab/ManiSkill (`stack_cube.py`, `peg_insertion_side.py`)
- RoboCasa 2406.02523 ; ARNOLD 2304.04321 ; Factor World 2307.03659 ; GemBench 2410.01345 ; VLABench 2412.18194 ; AGNOSTOS 2505.15660 ; RoboArena 2506.18123 ; SIMPLER 2405.05941 ; RoboTwin 2.0 2506.18088 ; FurnitureBench 2305.12821 ; ManipulationNet 2603.04363 (all arxiv.org/abs/<id>) ; KitchenShift https://github.com/etaoxing/kitchen-shift
- Bib checks: arXiv API (export.arxiv.org) for 2506.01844, 2603.08519, 2410.24164, 2501.15830/15835, 2503.22020, 2410.05260, 2403.03954, 2406.09246, 2405.12213, 2310.08864, 2303.04137, 2304.13705, 2212.06817, 2307.15818, 2108.03298, 2109.12098, 2006.04779, 2401.02117 ; Crossref DOIs 10.1109/LRA.2020.2974707, 10.15607/RSS.2023.XIX.026, .XIX.025, .XIX.016, .XIX.039, 10.15607/RSS.2024.XX.090, .XX.067, .XX.133, 10.15607/RSS.2022.XVIII.035, 10.15607/RSS.2025.XXI.010, 10.1109/iros.2012.6386109 ; PMLR v164 mandlekar22a, shridhar22a ; PMLR v270 kim25c ; Grasp'D https://arxiv.org/abs/2208.12250 (ECCV 2022) ; LeRobot https://github.com/huggingface/lerobot
- Menagerie: https://github.com/google-deepmind/mujoco_menagerie (folder `franka_emika_panda`, `LICENSE`)
