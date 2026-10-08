# AFFORD-Bench, RSS 2027 Stage 1: adversarial AC and reviewer audit

Date: 2026-10-08. Scope: `AFFORD-Bench_RSS'27/paper-template-latex/main.tex` (as of commit `72f9ae0`), `planning.md`, and the code it describes (`afford_pro/`, `scripts/`).

## Evidence base and reproduction conditions

The audit reads the code and runs it. All runs used MuJoCo 3.2.7 (Python 3.13), rendering disabled. The repository cannot run on any machine except the author's, so two substitutions were needed:

- every `afford_pro/assets/*.xml` sets `meshdir="D:/Selim Sarowar/Cursor_Projects/cap-x/..."`; the Franka Panda meshes from MuJoCo Menagerie (`franka_emika_panda/assets`, identical file names) were substituted;
- `kitchen_coffee.xml:134` and `tabletop_sorting.xml:134` load `D:/.../cartoon_only.png`; a builtin checker texture was substituted.

The substitutions touch only robot meshes and one texture. Object geometry, masses, friction, controller, oracle, and success logic are unchanged. Arm-object interaction could still differ from the author's machine, so every number below must be re-measured on the original assets with `scripts/sanity_baselines.py` before it enters the paper. Per-task results are in `docs/reviews/sanity_baselines_2026-10-08.csv`.

| Check (nominal suite, all 210 tasks) | Result in this reproduction | Paper claim |
| --- | --- | --- |
| Oracle success (deterministic, 1 episode) | 171/210 (81.4%); failures in D1 12, D4 10, D5 9, D6 8 | 100% on all 210 (Table II, Sec. VII-B) |
| Random policy, 5 episodes x 400 steps | SR > 0 on 11 tasks; 100% on 7 (017, 018, 019, 046, 193, 198, 209); aggregate 4.0% | 0.0% in every domain |
| Null policy (never moves, gripper open) | Solves 11 tasks: 017, 018, 019, 030, 046, 060, 090, 120, 193, 198, 209 | not reported |
| Success predicate already true at reset | 8 tasks: 017, 018, 019, 030, 046, 060, 090, 120 | not reported |
| Initial object pose across reset seeds | identical for 210/210 tasks | "initial pose jitter <= 1.0 cm" (L1) |
| Oracle trajectories across seeds | bit-identical actions and TCP poses (4 tasks checked) | 50 distinct demos per task |
| Simulated time per env step | 2 ms (200 steps = 0.400 s) | 20 Hz control, 500 Hz solver |

Oracle failure mechanism: the shaft (task 001), the 15 mL tube (091), and the dowel pin (016) topple and roll 6 to 9 cm under gravity within 0.4 s of reset, with the arm idle. The oracle computes all waypoints from the reset pose (`afford_pro/oracles/parametric_oracle.py:28`), so it grasps empty space. Every failing task uses a tall or thin body (shaft 14, conical tube 12, pin 6, microtube 6, gear 1).

## Bottom line (Area Chair)

As written, Stage 1 would be rejected, with high confidence. The title, abstract, Table I, Sec. IV, and H1/H2 rest on contact physics: friction, mass, compliance, clearances of 1.5 to 5 mm. In the released code a grasp welds the body to the TCP, a release within 4.5 to 6 cm teleports the body onto the goal, every tolerance is 30 to 40 mm, the physical suite runs as nominal, and nominal resets are deterministic. Table II's 100% oracle and 0% random are contradicted by a short reproducible check. Any one of these is enough for a reviewer to doubt the rest.

The benchmark is recoverable. The issues are implementation gaps, not a flawed concept. A Stage 1 that discloses what is implemented, fixes the cheap defects before 4 December, and reports measured sanity floors is a credible invitation-to-rebuttal paper. `main.tex` has been revised to that standard (Section 6); the code fixes are ranked in Section 7.

## 1. Reviewer reports

### Reviewer 1: manipulation mechanics and contact physics

Summary. The paper proposes 210 MuJoCo tasks framed as affordance-constrained MDPs, with physical perturbations of friction and mass.

Soundness, major:
1. No contact dynamics govern any success. `parametric_env.py:87-98` attaches the body to `tcp - 1 cm` with identity orientation whenever the gripper command is negative and the TCP is within 6.5 cm of the body origin. Finger contact plays no part. On release, `parametric_env.py:91` snaps the body to the goal pose if it is within `tolerance_xy + 2 cm`. The custom envs do the same with 4.5 to 6 cm radii (`shaft_assembly.py:83`). "Peg insertion" and "gear meshing" are therefore placement into a 30 to 40 mm disc followed by teleportation.
2. L4 does not exist. `PerturbationMode` (`engine.py:10`) has no physics mode, and `scripts/evaluate_ood.py` passes `p4_physical` straight through, so the physical suite is the nominal suite. Even if friction were perturbed, the weld would make it irrelevant during transport, and MuJoCo combines geom friction by taking the maximum by default, so the fingers' sliding friction of 3.0 would dominate an object set to 0.1.
3. The stated nominal friction μ0 = 0.70 appears nowhere. Object geoms use 1.2 to 1.8, the table 0.8, finger pads 3.0.
4. The control description is wrong. The arm is driven kinematically: IK writes `qpos` directly (`base_franka_env.py:193`), then one `mj_step` of 2 ms runs (`:207`). With an action scale of 0.04 m per step, commanded TCP speed reaches 20 m/s; the Panda's Cartesian limit is about 1.7 m/s [unverified]. "20 Hz Cartesian impedance" is not what the code does.
5. The rotation channels of the 7-D action are ignored (`base_franka_env.py:191`), and the TCP offset is a fixed world -z shift (`:106`), correct only when the hand points straight down.

Questions. What would change in any reported number if all contact were removed and objects were moved by script? How is "contact stability" in C_aff measured?

Score: strong reject as written; weak accept if contact grasping and L4 are implemented and validated on D1.

### Reviewer 2: foundation models and VLA evaluation

Summary. A policy-agnostic dataset and protocol for imitation and VLA policies, with chained-prompt domains (D5 to D7) aimed at subtask-aware models.

Evaluation, major:
1. Language grounding is broken for a third of the tasks. In 67 of 210 tasks the instruction names an object that is not the simulated body. Task 208 asks to "fill the silicone ice tray at the tap" while the env moves `blue_block`; task 169 asks for a QFP chip, simulated as a spur gear; task 004's bearing ring is also the spur gear. A VLA that ignores the instruction and moves the only movable body is scored as grounded.
2. The chains do not chain. Every D5 to D7 task evaluates one body, and the episode terminates when that body first reaches its goal. Under the subtask decomposition in `configs/prompts_augmented.yaml`, 57 of 90 chained tasks end before the last instructed step, often after step 1 (e.g., 121 ends after the shaft is seated, before the gear and pin). H3 cannot be tested, and AtomVLA-style subtask evaluation has nothing to score.
3. Format claims. OpenVLA and Octo consume RLDS, and SmolVLA consumes LeRobot datasets; Robomimic HDF5 is convertible to these, not directly compatible.
4. Two instructions are corrupted: tasks 147 and 149 contain TAB+`o` where `$\to$` was intended (`task_definitions.py:1947, 1973`; `\t` is a Python escape).
5. Citations are wrong: the SmolVLA entry has the wrong title and authors (it is Shukor et al., arXiv:2506.01844); the AtomVLA entry has the wrong title, authors, and year (it is Sun et al., arXiv:2603.08519, 2026); LIBERO-PRO is dated 2026 but was posted in October 2025 (arXiv:2510.03827). A reviewer who checks one of these will discount the rest.

Score: reject as written; the multi-tier prompt set (Task 1 deliverable) and flagged referents are a constructive start.

### Reviewer 3: benchmarks and experimental methodology

Evaluation, major:
1. No initial-state distribution. `apply_spatial_perturbation` returns the nominal pose unless the mode is spatial or combined (`engine.py:53`), so nominal resets are identical for every seed. The 50 demos per task are copies: 10,500 demonstrations contain 210 unique trajectories. "Evaluation seeds disjoint from training" means nothing for the nominal suite, and ID success of 100% after training would show only that the policy can replay one trajectory.
2. OOD sampling is neither seeded nor stationary. The perturbation RNG is constructed with `seed=None` and never re-seeded (`set_seed` is never called), so suites are not reproducible. `apply_visual_perturbation` adds to `model.light_pos` and multiplies `model.light_diffuse` in place on every reset (`engine.py:89-94`), as does object color (`:110`), so perturbation severity random-walks with the episode index.
3. Sanity floors are wrong (table above). Eight tasks succeed at reset. Seven are solved 100% by random actions.
4. Tolerances. Table I states clearances of ±1.5 to 20 mm and H2 relies on D1, D4, and D6 having ε ≤ 3 mm; all 210 TaskDefs use 40 mm / 30 mm, and the custom envs use 30 to 35 mm / 20 to 25 mm. H2 is untestable.
5. Claimed artifacts are missing: dense rewards (the reward is sparse, `base_franka_env.py:239`), FVR (no force monitor exists), depth in the dataset (`generate_all_150_tasks.py:53` never passes `render_depth=True`), camera intrinsics (not stored). Fig. 2's wrist-RGB row is black for D2 to D7; if the generator records the same frames, the wrist stream is empty for six domains.
6. Statistics. No training seeds, no confidence intervals, no learned baseline. BENCHMARK_RESULTS.md covers 9 tasks with 1 episode per split.
7. Missing prior work. COLOSSEUM (RSS 2024) already perturbs physical properties alongside 13 other factors. Factory (RSS 2022) and IndustReal (RSS 2023) simulate contact-rich insertion and gear meshing and transfer to a real Franka. The claim of being "the first systematic benchmark unifying ... affordance-space perturbation" fails against COLOSSEUM.

Score: reject as written; would move to borderline with fixes 1 to 6 of Section 7 and a measured Table II.

## 2. Notation inconsistencies in Section III (original manuscript)

1. 𝒯 is the task tuple and also, as 𝒯_p, the transition kernel. Fixed: task κ, kernel P.
2. ε_tol is a vector in Φ but bounds a scalar weighted norm ‖·‖_W ≤ ε_tol. The norm type of ‖·‖_W is undefined, and W = diag(w) duplicates what a vector tolerance would express. The code uses two scalars, ε_xy (2-D Euclidean) and ε_z (absolute). Fixed: ε = (ε_xy, ε_z), matching the code.
3. C_aff ⊂ S × A is a set of state-action pairs, but its constraints involve only state, and SR tests s_T ∈ C_aff, a state, which is a type error. Fixed: C_aff ⊂ S.
4. The text says "the terminal state trajectory must satisfy" the constraints, J uses ∃t ≤ H, and SR uses terminal s_T. These are three different success definitions. The code terminates at the first success (∃t). Fixed: ∃t everywhere.
5. F_friction / F_normal ≤ μ is Coulomb's law, enforced by the contact solver for every contact; as a task constraint it is vacuous. F_max is never given a value or a unit, and no code computes F_contact. Fixed: contact force is a monitored safety quantity for FVR, with F_max marked TODO.
6. The action is described as "delta velocity Δp" and "delta rotation Δθ". The code applies a position increment of 0.04·Δp m per step, ignores Δθ, and maps a_grip to a per-finger opening. Fixed.
7. R(s, a) ∈ {0, 1} is sparse in Sec. III-A, but Sec. III-B lists "dense step-level rewards". Fixed: sparse.
8. 𝐦 (bold) is a "mass distribution" in Φ, but L4 scales a scalar m_obj. Fixed: scalar m of the evaluated body.
9. r_0 is called an "initial kinematic configuration distribution", but it is a point mass in the code. Fixed: ρ0, with a TODO for L1 jitter.
10. "Transition probability" P(s'|s, a): MuJoCo is deterministic; stochasticity enters only through ρ0 and the perturbation sampler. Stated explicitly.
11. Gripper width is given as w ∈ [0, 0.08] m; the code observes per-finger opening in [0, 0.04] m (observation space bound 0.05). Fixed.
12. The TCP pose quaternion convention is unspecified, and the code's docstring says `[x, y, z, qx, qy, qz, qw]` while `mju_mat2Quat` returns w-first (`base_franka_env.py:109-113`). Fixed in the paper (w-first); fix the docstring.
13. D ∈ ℝ^{640×660} versus "640×660 RGB": MuJoCo's Renderer is built with height=640, width=660, so H × W = 640 × 660. Stated as H × W.
14. P = K⁻¹[u, v, 1]ᵀ D(u, v) needs K, which is not stored, and MuJoCo cameras look down −z with OpenGL axes, so a sign flip is required. Marked TODO: store K.
15. PRI = SR(P)/SR(Nominal) is undefined when nominal SR = 0. Domain restricted.

## 3. Overclaims about MuJoCo contact physics

| Location (original) | Claim | What the code does |
| --- | --- | --- |
| Abstract, Sec. I | physical affordances "that govern physical interaction" are evaluated | grasp is a kinematic weld; success is geometric |
| Fig. 1 | "20 Hz Cartesian Impedance" | DLS IK writes qpos; one 2 ms step per action |
| Sec. III-A | T_p "governed by MuJoCo multi-body physics" | the evaluated body's pose is overwritten every step while grasped and on release |
| Eq. (1) | contact stability F_contact ≤ F_max, friction cone | neither computed nor checked |
| Table I | clearances ±1.5 to 20 mm; "rigid peg-in-hole friction", "fragile glass contact", "dynamic toppling", "drawer opening", "pouring", "cable routing" | 30 to 40 mm everywhere; no articulated drawers, fluids, or cables exist in any scene |
| Sec. IV L1 | μ0 = 0.70, jitter ≤ 1 cm | object μ 1.2 to 1.8; zero jitter |
| Sec. IV L2 | ±6 cm, contracted insertion tolerances | ±3 cm (parametric) or ±4 cm (shaft env); tolerances unchanged; yaw discarded at grasp |
| Sec. IV L4 | grasp stability, slippage, compliance under μ ∈ [0.1, 1.2], m ∈ [0.5, 3] m0 | not implemented; suite equals nominal |
| Sec. V-A | "closed-loop algorithmic oracles", "minimum-jerk quintic splines", "verified using simulation contact manifolds" | open-loop waypoints from the reset pose, smoothstep (cubic) interpolation, geometric check only |
| Sec. VI-A | 500 Hz solver, 20 Hz closed loop | 500 Hz solver, but 500 Hz actions |
| Sec. VI-C | FVR "episodes terminating due to contact collisions or excessive forces" | no such termination exists |
| Sec. VII-B | 100% oracle across 210 tasks on disjoint seeds; random 0% | 81.4% and 4.0% in this reproduction; seeds are irrelevant in nominal |
| planning.md, vulnerability 1 | "docking, peg insertion, gear meshing ... fully governed by MuJoCo contact physics" | false; release snaps the body to the goal |

The planning.md defense, "kinematic attachment for transport, contact physics for insertion", would not survive a reviewer who reads `parametric_env.py`. Do not use it in a rebuttal.

## 4. The five hardest Stage 1 metareview questions

**Q1. Where is the contact physics?** "Grasps are welds and goals snap within centimetres. Which reported quantity depends on friction, mass, or clearance, and how would H1 or H2 be falsified?"
Answer only after fix F1 (contact grasp for at least D1 and D4) and F2 (L4 implemented). Then show one curve: oracle success on D1 versus μ and versus ε_xy, demonstrating that the axis changes outcomes. Without that curve, concede that Stage 1 measures goal-reaching under spatial and visual shift, and move contact to the charter. The revised manuscript already concedes this in Sec. IV "Implementation status".

**Q2. What is the initial-state distribution, and how many independent demonstrations exist?** "Nominal resets are identical, so the 50 demos per task are copies. What does ID success measure? Are disjoint seeds meaningful?"
Answer after fix F3 (L1 jitter, seeded RNG, regenerated demos). Report unique-trajectory count and the per-task pose distribution. Concede that the Stage 1 dataset statistics previously overstated diversity.

**Q3. Do the sanity floors hold?** "Random and null policies solve several tasks; Table II says 0% and 100%. Which tasks were removed, and what protocol produced Table II?"
Answer with Table II measured by `scripts/sanity_baselines.py` on the original assets, the list of tasks fixed or excluded (at least the 11 null-solvable tasks), and the oracle failures explained (toppling spawn poses) and fixed (F4).

**Q4. Does the language mean anything?** "67 instructions name absent objects; D5 to D7 terminate after one body. How do you evaluate VLA grounding and H3?"
Answer: `configs/prompts_augmented.yaml` flags `referent_in_scene` and `instruction_covered_by_success` per task, and the paper restricts language-grounding analysis to flagged-valid tasks. For H3, implement multi-body sequential predicates for at least 10 D6 and 10 D7 tasks (F6) before 4 December, or move H3 entirely into the charter.

**Q5. What is new relative to COLOSSEUM, LIBERO-PRO/X, and IndustReal, and where are the learned baselines?** "COLOSSEUM perturbs physical properties; IndustReal does sim-to-real insertion and gear meshing. Stage 1 has no trained policy."
Answer: the differentiator is tolerance-indexed tasks (H2), meaning the same task at ε_xy ∈ {5, 10, 20, 40} mm, across seven domains with policy-agnostic data. Neither COLOSSEUM nor LIBERO-X indexes tasks by tolerance. Support it with BC and Diffusion Policy on D1 and D3 (3 training seeds, 50 episodes per task, Wilson intervals) in Table II.

## 5. Rebuttal strategy

The rebuttal window is one week (5 to 12 February 2027). Nothing that requires new engineering can be done inside it, so the defensible rebuttal must already be in the submission:

1. Concede first, in one sentence each, what Stage 1 does not do (contact grasping, L4, multi-body chains) and point to the "Implementation status" paragraph that disclosed it. Reviewers punish discovered overclaims far more than disclosed limitations.
2. Answer with numbers measured on the submitted code: sanity floors, oracle solvability after spawn fixes, and one tolerance sweep. The rebuttal should quote numbers, not intentions.
3. Shape the charter. RSS Stage 2 is an AC-issued Extension Charter, so propose its content yourself: contact grasping and L4 validated on D1/D4, multi-body predicates for D5 to D7, full baseline matrix, hardware pilot. A concrete, scoped charter is easier for an AC to adopt than an open promise.
4. Answer the strongest reading of each critique. For example, Q1's strongest form is "the axis is unfalsifiable"; answer with the falsification curve, not with a description of MuJoCo's capabilities.
5. Do not argue novelty against COLOSSEUM or IndustReal. Cite them as the standard and position AFFORD-Bench as the tolerance-indexed, cross-domain complement.

## 6. Revisions applied to `main.tex` in this change

- Abstract and Sec. I rewritten to claim only what the code supports, with a fourth contribution for the multi-tier language set. Em dashes removed.
- Related work: COLOSSEUM, Factory, IndustReal added. The unsupported "first systematic benchmark" sentence removed.
- Sec. III-A rewritten with consistent notation (Section 2 above).
- Sec. III-B: dense rewards, depth, and gripper range corrected; TODO to store K and enable depth.
- Table I: clearance column replaced by the real success tolerances; representative tasks limited to those that exist; evaluated bodies listed (16 rigid bodies).
- Sec. III-C: D5 to D7 single-body evaluation disclosed, with the 57/90 early-termination count.
- Sec. IV: L2 values corrected (±3 cm); L1, L3, and L4 TODOs; new "Implementation status" paragraph.
- Sec. V: oracles described as scripted open-loop waypoint oracles; identical-demo TODO; RLDS/LeRobot conversion stated.
- Sec. VI: control rate stated truthfully with a TODO for 25 substeps; seeds' scope clarified; Wilson intervals and 3 training seeds committed; PRI domain restricted.
- Sec. VII: hypotheses reworded to be falsifiable (H2 tolerance sweep, H3 product-of-step-rates test). Table II values replaced by TBD and a Null column added.
- Sec. VIII-A: new subsection "Physical Hardware Correspondence and Sim-to-Real Feasibility" (Task 3). Charter updated to peg insertion and gear meshing.
- `references.bib`: SmolVLA, AtomVLA, LIBERO-PRO corrected; COLOSSEUM, Factory, IndustReal, LeRobot added. The LeRobot author list follows the repository's suggested citation [unverified; check against the current README].

The compiled PDF is 6 pages: 4 content pages, with references on pages 5 and 6, and no overfull boxes (pdflatex, TeX Live 2023). Search for `\TODO` and `\TBD` before freezing the submission; 9 `\TODO` sites and the Table II cells remain.

## 7. Code fixes ranked by impact to effort (target: before 4 December)

| # | Fix | Impact | Effort |
| --- | --- | --- | --- |
| F0 | Ship assets: relative `meshdir`, vendor the Panda meshes and `cartoon_only.png` (or Menagerie as a submodule) | without it, no reviewer can run anything | small |
| F3 | L1 jitter in nominal; seed `PerturbationEngine` from `reset(seed)`; restore nominal lights/colors before each sample; regenerate demos | makes ID/OOD and dataset size meaningful | small |
| F4 | Fix spawn poses (settle objects for N steps at reset, or spawn at rest heights), read the object pose after settling in the oracle, and re-plan the grasp from the current pose | oracle solvability; removes 39 failures here | small |
| F5 | Remove or repair trivially solved tasks (goal = start, goal = home pose, reachable without motion) | sanity floors | small |
| F7 | 25 substeps per action (20 Hz) and a TCP speed cap; use position actuators (`data.ctrl`) instead of writing qpos | physical plausibility; prerequisite for any contact claim | small to medium |
| F8 | Record depth and intrinsics; add a wrist-RGB mean-intensity check to `verify_dataset_integrity.py` | substantiates the 3D and dual-camera claims | small |
| F2 | L4: per-episode `model.geom_friction` (object and pads) and `model.body_mass` scaling, restored after each episode | enables H1 | small; meaningful only with F1 |
| F1 | Contact grasp: delete the weld and snap, rely on finger friction, and add real clearances (bore radius = part radius + ε) for D1/D4 insertion | the paper's central claim | medium to large; start with tasks 001 to 003 and 091 to 092 |
| F9 | Per-task tolerances in `TaskDef` plus tolerance-indexed variants | enables H2 | small once F1 lands |
| F6 | Multi-body sequential predicates for D5 to D7; remove or re-assign the 67 alias instructions | enables H3 and VLA grounding claims | medium |
| F10 | Fix `\to` escapes (tasks 147, 149), the quaternion docstring, and the TCP offset rotation | correctness | trivial |

Order of work: F0, F3, F4, F5, F7, F8 first (all small), then measure Table II and train BC and DP on D1 and D3, then F1 and F2 on D1, then F6. If F1 slips, submit with the current honest framing; it is defensible.

## 8. Notes on the other two deliverables

- `configs/prompts_augmented.yaml`: four tiers for all 210 tasks, with simulator grounding per task. The coordinates in the geometric tier are generated from `task_definitions.py`, and `tests/test_prompts_augmented.py` checks them (633 tests). The affordance tier conditions on contact properties the simulator does not model yet, so until F1 and F2 land it tests sensitivity to wording only. That caveat is in the file header.
- Sec. VIII-A (hardware): written to avoid the "1:1 aligned" claim. Robomimic HDF5 is not RLDS or the LeRobot format; 640 × 660 is not a native RealSense resolution; ALOHA, SO-101, and Piper are joint-space embodiments while the simulator stores Cartesian increments. The section states these gaps and designs the pilot around them. It assumes access to Franka, SO-101, and Piper arms with RealSense cameras, as stated in the request; that is not verified here.
