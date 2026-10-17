# Progress

Last updated: 2026-10-17, end of session 1 (stopped at the usage limit). Verdict: **NOT YET SUBMISSION-READY**
(no learned-policy results; L4 range uncalibrated; author decisions D1-D8 open; nothing committed).

## Done this session (evidence in EXPERIMENT_LOG.md E-001..E-011, findings A-01..A-28)
- Audit of code, data, paper, prior reviews; literature comparison (`LITERATURE_COMPARISON.md`, spot-checked).
- Measurements on the current physics (MuJoCo 3.5.0): sanity floors (E-004), runtime task audit (E-005), validation
  of all 10,500 episodes with replay (E-007), L4 friction x mass sweep (E-008), force probe (E-009), reactive probe
  (E-010), one-task baseline pipeline check (E-011).
- New code: `afford_pro/{provenance.py, task_tiers.py}`, `afford_pro/datasets/{validation.py, replay.py,
  generation.py}`, `afford_pro/envs/headless.py`, `afford_pro/evaluation/{stats.py, rollout.py}`,
  `afford_pro/learning/{features.py, policies.py}`, package `__init__.py` files; scripts `validate_datasets.py`,
  `audit_tasks.py`, `aggregate_sanity.py`, `l4_sweep.py`, `force_probe.py`, `evaluate_policy_suite.py`.
- Rewritten: `hdf5_writer.py` (atomic, provenance, final state, canonical instruction), `generate_all_150_tasks.py`,
  `generate_demos.py`, `train_policy.py` (lowdim modality, normalizer, validation selection), `evaluate_id.py`,
  `evaluate_ood.py` (wrappers), `sanity_baselines.py` (per-episode log, H, provenance), `setup.py` (package data,
  mujoco==3.5.0, wheel verified in a temp venv), `requirements-ci.txt`, `.gitignore`, `README.md`, `docs/index.html`.
- Tests added: `test_dataset_validation.py`, `test_generation.py`, `test_task_registry.py`.
- Paper: abstract, intro, related work, Table I caption, tiers paragraph, Sec. IV L1/L3/L4, dataset section,
  Table II, calibration text, Sec. VIII condensed; 13 bib entries corrected or added; builds at 6 pages (5 + refs),
  no undefined references; `main.pdf` replaced.
- Quarantined synthetic results and the withdrawn comparison figure (`docs/quarantine/synthetic_projections/`).
- Pre-session uncommitted user changes preserved; their diffs saved in `docs/roadmap/backups/`.

## Session 2 (2026-10-18)
- Code review (code-reviewer agent): no CRITICAL/HIGH; 13 MEDIUM/LOW findings, all addressed except long legacy
  functions in `hdf5_writer.py` and the global rendering patch in `sanity_baselines.py`. Changes: generator validates
  the per-process `.partial` before naming it (`.invalid`/`.conflict`/`.incomplete` otherwise) and exits non-zero on
  failure; validator clamps frames, treats a changed scene XML as an error, checks the recorded generation status and
  seed range, and turns malformed files into INVALID rows; training refuses missing datasets; evaluation refuses
  legacy checkpoints and guards seed ranges; tiers raise on tasks missing from the YAML.
- A-29 fixed: perturbation suites were unpaired by seed; model perturbations now use their own stream.
- A-26 root cause found and fixed: recorded actions were float32 roundings of executed float64 actions.
- New tests: `test_perturbation_isolation.py`, `test_replay.py`, `test_evaluation_learning.py`, conflict test.
  Full suite: 2,403 passed.
- Fig. 2 regenerated (D4 wrist view, labels); NOTICE lists verified Menagerie modifications; `docs/DATASET_CARD.md`.
- E-013 (re-measure oracle/L4/force on the fixed code) running; paper Table II, Sec. IV sweep numbers, and force
  numbers must be updated from `sanity_2026-10-18/`, `l4_sweep_2026-10-18.csv`, `force_probe_2026-10-18.csv`.

## Decisions required from the authors
- D1 Headline scope: present the 66-task validated core as the benchmark (recommended) or 210 with tiers (current text
  does the latter with tier disclosure).
- D2 Demos longer than the env limit (114): drop/regenerate (recommended, generator now enforces it) or raise H.
- D3 Install a CUDA build of torch for the GTX 1080 Ti (pip, user env). D4 Baseline training campaign (reference
  Diffusion Policy/ACT via LeRobot + BC, 66 core tasks, 3 seeds, 5 suites).
- D5 Licence for AFFORD-Bench code and data. D6 Commit/push the session's work.
- D7 Recalibrate L4 from the sweep (e.g. mu in [0.03, 0.06], mass x[2.5, 5]); requires regenerating L4 evaluations.
- D8 GitHub Pages is public and serves the old page with fabricated results: push the corrected page or disable Pages
  now (A-25). Also: archive (18 GB) and old checkpoints may be deleted only with approval.

## Open technical items
- Replay divergence root cause (A-26; not the warm start).
- Perturbation-isolation tests (C-21); fixed-start and unmarked-goal subsets reported separately in results.
- Regenerate Fig. 2 (black D4 wrist image, stale camera); recheck `docs/FRONTIER_COMPLEX_TASKS.md`, `planning.md`,
  `BENCHMARK_RESULTS.md` banners; diff robot XML against Menagerie `panda.xml` for the NOTICE.
- Code review of the new modules (user rule: code-reviewer agent) was not run before the stop.

## Next three actions
1. Run the full test suite (`python -m pytest -q tests`) and a code review of the new modules; fix findings.
2. Get decisions D1, D6, D8 (public page) from the authors.
3. With D3/D4: train BC + reference DP on the core tier (3 seeds), evaluate 5 suites with `evaluate_policy_suite.py`.
