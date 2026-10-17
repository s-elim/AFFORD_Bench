# AFFORD-PRO-150 Benchmark Evaluation Report (Historical Smoke Test)

> **[AUDIT PROVENANCE NOTE]**: This file records a legacy 9-task, 1-episode-per-split (`n=1`) oracle smoke test generated prior to the October 8, 2026 physics hardening (`cba995e`) and excludes the `p4_physical` perturbation split. For the verified post-hardening 210-task calibration (5 seeds/task, including `p4_physical`), see [`docs/reviews/sanity_baselines_2026-10-08.csv`](docs/reviews/sanity_baselines_2026-10-08.csv) and Table II of [`AFFORD-Bench_RSS'27/paper-template-latex/main.tex`](AFFORD-Bench_RSS'27/paper-template-latex/main.tex).

- **Evaluated Policy**: `ORACLE`
- **Total Evaluated Tasks**: `9`
- **Episodes per Split**: `1`
- **Execution Duration**: `134.15s`

## Out-of-Distribution Generalization Performance

| Task ID | Domain | Nominal (ID) | P1 (Object) | P2 (Spatial) | P3 (Visual) | Combined OOD | OOD Avg | GDI (%) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `industrial_shaft_assembly` | Industrial & Manufacturing | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `industrial_gear_mesh` | Industrial & Manufacturing | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `industrial_pin_insertion` | Industrial & Manufacturing | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `kitchen_patty_stack` | Culinary & Kitchen Operations | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `kitchen_cheese_place` | Culinary & Kitchen Operations | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `kitchen_mug_to_espresso` | Culinary & Kitchen Operations | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `tabletop_sort_red` | Tabletop & Domestic | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `tabletop_sort_green` | Tabletop & Domestic | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `tabletop_sort_blue` | Tabletop & Domestic | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MACRO AVERAGE** | **All Evaluated Tasks** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** |

*Generalization Degradation Index (GDI): $\text{GDI} = \frac{\text{ID}_{\text{SR}} - \text{OOD}_{\text{SR}}}{\text{ID}_{\text{SR}}} \times 100\%$ (lower is better)*
