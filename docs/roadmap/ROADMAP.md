# AFFORD-Bench roadmap to an RSS 2027 submission

Deadline: Stage 1 extended abstract, 4 December 2026 (AoE). Today: 2026-10-17 (48 days).
Goal: a benchmark whose validated task inventory, dataset, evaluation protocol, and paper agree, and whose every
number traces to an artifact. Task count, dataset size, and paper length are not goals.

Related files: `AUDIT_REPORT.md` (findings A-xx), `CLAIM_EVIDENCE_LEDGER.md` (claims C-xx), `EXPERIMENT_LOG.md`
(runs E-xxx), `PROGRESS.md` (current state), `LITERATURE_COMPARISON.md` (prior work).

## Autonomy and approval boundaries (working assumptions)
- Done without asking: reading, auditing, editing source/tests/docs, local CPU runs up to about 2 h wall clock that
  only read datasets or write new files under `docs/roadmap/evidence/` or the scratchpad.
- Needs explicit approval: deleting, moving, overwriting, or rewriting any dataset or checkpoint (including HDF5
  attribute edits); dataset regeneration (about 12 to 24 CPU-hours, 450 GB); policy-training campaigns; installing a
  CUDA build of torch; choosing a licence; committing, pushing, or publishing; narrowing the paper's headline scope.

## Critical path
```
T0 source of truth ──> T1 re-measure sanity floors ──┐
                     T2 task-semantics audit + tiers ─┼─> T6 paper/claims sync ──> T8 Stage 1 freeze
                     T3 episode-level dataset validation ─┤                         ^
                     T4 generator provenance + atomic ────┘                         |
                     T5 baselines (pipeline fix -> approval -> runs) ───────────────┘
                     T7 literature comparison ──────> T6
```

## Status by workstream (2026-10-18)
| Workstream | State |
| --- | --- |
| T0 source of truth | done |
| T1 sanity floors on current physics | done (E-004), re-measured after the pairing and float32 fixes (E-013, running) |
| T2 task semantics and tiers | done: 66 core / 35 reach / 109 mismatch; audited subsets; decision D1 open |
| T3 episode-level validation | done for all 210 files (E-007); replay cause found and fixed (E-012) |
| T4 generator provenance and safe writes | done and checked end to end (E-014); regeneration needs approval (D2) |
| T5 baselines | pipeline fixed and checked on one task (E-011); campaign needs approval (D3, D4) |
| T6 paper | synchronized with evidence through E-012; Table II and L4/force numbers to refresh from E-013 |
| T7 literature | done (`LITERATURE_COMPARISON.md`), bib corrected |
| T8 release | packaging, CI pin, NOTICE, dataset card done; licence (D5), public page (D8), commit (D6) open |

## Workstreams

### T0 Establish source of truth [MANDATORY, done]
Acceptance: repository state, environment, datasets, and discrepancies recorded (`AUDIT_REPORT.md`, `PROGRESS.md`).

### T1 Re-measure Table II on the current physics [MANDATORY, blocker for the paper]
- Oracle (nominal, L4), random, null on all 210 tasks, 5 seeds; record success within H=400 and within the oracle
  budget (600); provenance sidecar. Script: `scripts/sanity_baselines.py` (extended 2026-10-17).
- Acceptance: CSVs + provenance in `docs/roadmap/evidence/sanity_2026-10-17/`; aggregate with Wilson intervals
  (task-clustered where stated); paper Table II replaced by these numbers or marked stale.

### T2 Task-semantics audit and task tiers [MANDATORY, blocker]
- Static registry checks as tests: unique ids/numbers, domain counts, tolerance tiers, object exists in scene XML,
  instruction escapes, referent/coverage flags consistent with task type, reach tasks flagged.
- Runtime audit per task (no rendering): goal calibration (resting vs hold goal, drift, tilt), start fallback rate,
  graspability, success at reset, reach-target distance to the named object, distractor placement.
- Tiers (non-destructive, a manifest plus `afford_pro.list_tasks(tier=...)`):
  `core` = manipulation tasks whose referent is simulated and whose success covers the instruction;
  `reach` = fixed-point reach tasks (reported separately, never mixed into manipulation success rates);
  `instruction_mismatch` = referent not simulated or success covers only part of the instruction.
- Acceptance: tests pass; `evidence/task_audit_*.json`; tier manifest; counts in the paper match the manifest.
- Decision for the authors (D1): headline the validated core (about 80 tasks, see T2 output) or keep 210 with tiers.

### T3 Episode-level dataset validation [MANDATORY, blocker]
- Validator module `afford_pro/datasets/validation.py` + CLI `scripts/validate_datasets.py`: checks every episode
  (schema, shapes, dtypes, finiteness, per-step lengths, terminal reward/done semantics, `init_state == states[0]`,
  obs/state alignment, `next_obs` shift, seeds unique, duplicate initial states and action sequences, split masks,
  registry consistency, episode length vs env horizon and H, model_file vs current scene XML); image checks on
  sampled frames; optional deterministic replay of recorded actions.
- Diversity and coverage statistics: start-pose spread of the evaluated body, episode lengths, action statistics,
  instruction-variant counts, inferred generation attempts.
- Machine-readable report + markdown summary + manifest (per-file sha256 optional).
- Acceptance: all 210 files checked at episode level; replay on a stratified sample; every file VALID,
  VALID_WITH_WARNINGS, or INVALID with reasons; generator skip logic uses the strict validator.

### T4 Generator provenance and safe writes [MANDATORY]
- Write to `<name>.partial`, validate, then atomic rename; record `collect_provenance()`, attempts, failed seeds,
  oracle budget, and H in the file; respect the env horizon (decision D2: drop demos > H, or raise H consistently).
- Acceptance: unit tests for interrupted writes and provenance attributes; one small task regenerated into a
  scratch directory and validated.

### T5 Baselines [MANDATORY for Stage 1 numbers; runs need approval]
- Fix pipeline: object pose (from `states`) in the low-dim modality, normalization statistics stored in the
  checkpoint, validation-loss checkpoint selection, seeded evaluation, per-episode result logs with provenance,
  Wilson intervals, multiple training seeds.
- Pipeline validation (allowed): one task, few epochs, CPU, 5 evaluation episodes, to prove the loop runs end to end.
- Campaign (approval D3/D4): BC and Diffusion Policy on the core tiers of D1 and D3, 3 training seeds, 50 evaluation
  episodes per task per suite. Cost estimate in PROGRESS once the pipeline is validated.

### T6 Paper and claims synchronization [MANDATORY]
- Every number in `main.tex` maps to a ledger row with status SUPPORTED, or is removed / marked as design.
- Fix: Table II, abstract, Sec. IV scatter schedule (now 1/20 step too), L3 clipping, L4 slip claims, dataset size and
  mean episode length, MuJoCo version, Fig. 2, task-tier counts, related-work facts (from T7).
- Acceptance: no `\TODO`/`\TBD` left; ledger has no UNVERIFIED rows referenced by the paper; PDF builds within 6 pages.

### T7 Literature comparison [MANDATORY]
- Primary-source comparison table for LIBERO, LIBERO-PRO, LIBERO-X, COLOSSEUM, LIBERO-Plus, Factory/IndustReal and
  others; verdicts for every competitor statement; bib entries verified (`LITERATURE_COMPARISON.md`).
- Acceptance: figure `benchmark_comparison` corrected or withdrawn; related work cites only verified facts.

### T8 Release readiness [MANDATORY before public release; partly author decisions]
- LICENSE (D5), third-party notices (Menagerie meshes, textures), dataset card (schema, provenance, limitations),
  pinned environment (MuJoCo version used for the data), CI smoke suite including validator and registry checks,
  quarantine of synthetic results, README rewritten to match code.

### T9 All 210 tasks valid end to end [MANDATORY, author decision 2026-10-18]
The authors require every one of the 210 tasks to be accurate, with the instruction-scene-predicate consistency of
LIBERO, LIBERO-PRO, LIBERO-X, and LIBERO-Plus (BDDL-style: named objects exist, the goal is a conjunction of
predicates over them, success = all predicates hold).

**Definition of a valid task (acceptance gate, checked by tests and the runtime audit):**
1. Every object and fixture the instruction names is a simulated body in the scene, identifiable from the image
   (color, shape, or position words that hold for every reset).
2. The goal is a conjunction of physical predicates over named bodies: *placed* (released, supported by the **named**
   receptacle or support body, at rest, upright within the tilt bound, within the position tolerance), *held* (both
   fingers, at the goal), or *actuated* (a mechanism joint past its threshold, e.g. a sprung button pressed, a drawer
   opened). The episode succeeds the first time all predicates hold at once.
3. Positions on featureless surfaces are marked by a visible target (a flat, non-colliding marker) or described
   relative to a visible body.
4. Every body that the goal moves starts from a randomized pose (start s.d. >= 1 cm unless it is seated in a fixture,
   which the instruction states).
5. Oracle >= 95% within H, null and random 0%, no success at reset (5+ seeds per task, MuJoCo 3.5.0).
6. The four instruction variants are rewritten to match: no nominal start coordinates, no unmodeled properties.

**Work breakdown (in dependency order):**
- T9.1 Goal schema: `TaskDef.goals` = list of placed/held/actuated predicates (single-body tasks become one-element
  lists); named support bodies; tests.
- T9.2 Environment: multi-body success, support-body check, joint predicates, per-body start randomization.
- T9.3 Oracle: executes a goal list in order (pick-place per body; press/push for mechanisms), with the existing
  grasp planning, retries, and release checks.
- T9.4 Assets: sprung buttons and other mechanisms the instructions need (e-stop, brew, centrifuge START, clamp,
  plunger, drawer), visible goal markers.
- T9.5 Redefine the 144 non-core tasks domain by domain (D1-D4 mismatches and reach tasks, then D5-D7 chains as
  2-3-predicate conjunctions over the bodies each scene actually contains); keep the 66 core tasks, adding named
  support bodies.
- T9.6 Validate every task against the gate (audit + oracle/null/random), iterate until all 210 pass.
- T9.7 Regenerate `configs/prompts_augmented.yaml` from the new definitions.
- T9.8 Regenerate all 10,500 demonstrations into a new folder (`datasets_v2/`); old data kept until the authors
  delete it. Validate every episode; replay must be exact.
- T9.9 Update paper, page, dataset card, and the leaderboard references; GPU baselines run on the authors' server.

## Optional / Stage 2 (not on the Stage 1 critical path)
- Multi-body sequential predicates for D5 to D7 (H3), tolerance-indexed variants (H2), Domain 8 validation,
  VLA evaluations, hardware pilot.

## Release gates (all must hold before "submission-ready")
1. Validated task inventory = code = data = paper counts.  2. Core instructions and predicates correct.
3. Every episode passes integrity checks; provenance documented (gaps disclosed).  4. Coverage and limitations
documented.  5. Stale or partial data cannot pass the generator's skip check.  6. Baseline numbers come from logged
evaluations.  7. Main experiments reproducible from documented commands.  8. Fair comparison with prior work.
9. No placeholders or unsupported numbers in the paper.  10. External researcher can install, validate, evaluate.
11. Limitations and pending experiments disclosed.  12. All mandatory tests pass.

## Risks
- R1: the replay check fails under the current code -> datasets would not match the released simulator; requires
  regeneration (D2) before any baseline.
- R2: the validated core is small enough to change the paper's framing (D1).
- R3: CPU-only training limits baselines to low-dim and small visual models before the deadline unless a CUDA torch
  is installed (D3).
- R4: MuJoCo version drift between CI (3.2.7) and data (3.5.0).
