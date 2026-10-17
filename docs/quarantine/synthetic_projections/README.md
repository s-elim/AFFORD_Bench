# Quarantined: not experimental results

Moved here on 2026-10-17 (see `docs/roadmap/AUDIT_REPORT.md`, findings A-02 and A-03). Nothing in this folder may be
cited, plotted, or reported as a measurement.

- `MODEL_BENCHMARK_EVALUATION.md`, `vla_model_benchmark_comparison_2026-10-08.{csv,json}`,
  `figures/model_performance_comparison*`: scores for pi0/pi0.5, SpatialVLA, AtomVLA, CoT-VLA, Diffusion Policy, ACT,
  OpenVLA, and SmolVLA were written by hand into the `BENCHMARK_PROFILES` dictionary of
  `scripts/evaluate_vla_baselines.py`. No model was trained or rolled out. The oracle/random/null rows came from the
  2026-10-08 sanity run, which has since been superseded (`docs/roadmap/evidence/sanity_2026-10-17/`).
- `figures/benchmark_comparison*` (from `scripts/generate_benchmark_comparison_figure.py`): states properties of
  LIBERO-PRO and LIBERO-X ("kinematic welds", "fixed mu = 0.5", "1x RGB 128x128", "VLA models only") that their
  papers and code contradict. Verdicts and sources: `docs/roadmap/LITERATURE_COMPARISON.md`.

Real evaluations use `scripts/evaluate_policy_suite.py`, which logs every episode with its seed, the checkpoint hash,
and the code provenance.
