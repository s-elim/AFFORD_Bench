# AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning

AFFORD-Bench is a MuJoCo benchmark for robot manipulation policies on a simulated Franka Emika Panda. Success is a
physical predicate on the settled state: the released body must be supported, at rest, upright, and within 2-2.5 mm
(precision domains) or 8 mm of its goal, after having been lifted. Grasping is frictional; there is no attachment,
snapping, or teleport. Demonstrations come from a privileged closed-loop scripted oracle and are stored as
Robomimic HDF5 with converters to LeRobot and RLDS.

Target venue: RSS 2027 (Stage 1 extended abstract, `AFFORD-Bench_RSS'27/paper-template-latex/`).

## Status (2026-10-17)

| Item | State | Evidence |
| --- | --- | --- |
| Task inventory | 210 registered tasks in 7 domains; **66 validated core tasks** (D1-D4), 35 fixed-point reach tasks, 109 instruction mismatches (all D5-D7 chains included) | `afford_pro/task_tiers.py`, `docs/roadmap/evidence/task_audit_2026-10-17.json` |
| Dataset | 210 files x 50 demos = 10,500 demos, 1,230,842 steps; every episode validated (0 errors, documented warnings) | `docs/roadmap/evidence/dataset_validation_2026-10-17/` |
| Sanity floors | Oracle 95.7% nominal, 95.0% L4; random 0/1050; null 0/210 | `docs/roadmap/evidence/sanity_2026-10-18/summary.md` |
| L4 physical suite | At the specified range (mu 0.12-0.25, mass x1.8-2.5) it does not lower oracle success; lower friction and heavier bodies do (16% at mu 0.02, x5). Recalibration pending | `docs/roadmap/evidence/l4_sweep_2026-10-18.csv` |
| Learned baselines | Not reported yet. Pipeline check: low-dim BC 16/20 on one task; the in-repo diffusion model does not learn (use a reference implementation) | `docs/roadmap/evidence/pipeline_check_2026-10-17/` |
| Licence | MIT (`LICENSE`), code and data. Robot meshes: Apache-2.0 (MuJoCo Menagerie), see `afford_pro/assets/robots/NOTICE.md` | - |

The working plan, findings, and claim-by-claim evidence are in `docs/roadmap/` (`ROADMAP.md`, `PROGRESS.md`,
`AUDIT_REPORT.md`, `CLAIM_EVIDENCE_LEDGER.md`, `EXPERIMENT_LOG.md`, `LITERATURE_COMPARISON.md`). Files under
`docs/quarantine/` are not experimental results.

## Installation

```bash
conda create -n afford_bench python=3.10 -y && conda activate afford_bench
pip install -e .            # pins mujoco==3.5.0, the build used for all data and numbers
pip install torch           # only for training/evaluating learned policies
python -m pytest -q tests   # about 2,400 tests, a few minutes on CPU
```

All reported numbers use MuJoCo 3.5.0 and the Panda meshes in `afford_pro/assets/robots/assets/`. Results with other
MuJoCo builds can differ, because contact resolution changes between releases.

## Tasks and tiers

| Domain | Tasks | Core / reach / mismatch | Tolerance (xy, tilt) |
| --- | --- | --- | --- |
| D1 Industrial precision | 001-030 | 17 / 6 / 7 | 2 mm, 5 deg |
| D2 Culinary operations | 031-060 | 20 / 7 / 3 | 8 mm, 15 deg |
| D3 Domestic tabletop | 061-090 | 17 / 8 / 5 | 8 mm, 15 deg |
| D4 Laboratory | 091-120 | 12 / 14 / 4 | 2 mm, 5 deg |
| D5 Long-horizon | 121-150 | 0 / 0 / 30 | scene tier |
| D6 Precision chains | 151-180 | 0 / 0 / 30 | 2.5 mm, 5 deg |
| D7 Service chains | 181-210 | 0 / 0 / 30 | 8 mm, 15 deg |

- **core**: the instruction names the simulated body and the success predicate covers the whole instruction.
- **reach**: the fingertip center must stay within tolerance of a fixed world point for 5 steps; instructions that
  mention wiping, pressing, or inspecting describe actions that are not simulated. Report separately.
- **instruction_mismatch**: the named object is not the simulated body, the chain ends after its first evaluated
  body, or the predicate contradicts the instruction (`afford_pro.task_tiers.mismatch_reason(task_id)`).

```python
import afford_pro
from afford_pro.task_tiers import tasks_in_tier, mismatch_reason
core = tasks_in_tier("core")                    # 66 task ids
env = afford_pro.make(core[0], perturbation_mode="nominal")
```

Domain 8 (tasks 211-240, `afford_pro.list_frontier_tasks()`) is a Stage 2 extension and has not been validated.

## Perturbation suites (`afford_pro/perturbations/engine.py`)

| Suite | What changes |
| --- | --- |
| `nominal` (L1) | Start of the evaluated body U(+-7 cm) per axis, yaw U(+-45 deg), roll/pitch U(+-10 deg) resolved by settling; TCP jitter +-1.5 cm. Rejection-sampled, shrinking to 1/2, 1/4, 1/10, 1/20 of the range; in 29 tasks the start varies by < 1 cm |
| `p1_object` | Color of the evaluated body x U(0.75, 1.25) |
| `p2_spatial` (L2) | Start offset radius U(7, 12) cm in a random direction, yaw magnitude U(45, 90) deg |
| `p3_visual` (L3) | Light position +-0.5 m, diffuse x U(0.5, 1.5) clipped at 1 (the key light brightens by at most x1.11) |
| `p4_physical` (L4) | Sliding friction U(0.12, 0.25) on the body and finger pads; mass and inertia x U(1.8, 2.5) |
| `combined_ood` (L5) | L2 + L3 + L4 + color |

Nominal model parameters are restored and the sampler is re-seeded at every `reset(seed)`.

## Data generation and validation

```bash
# Generate (skips valid files, refuses to overwrite invalid ones, records provenance, writes atomically)
python scripts/generate_all_150_tasks.py --domain industrial --num-demos 50
python scripts/generate_demos.py --task tabletop_sort_red --num-demos 5 --output-dir datasets/scratch

# Validate every episode (+ replay 3 demos per file under the current code)
python scripts/validate_datasets.py --replay-per-file 3 --workers 8 --out-dir docs/roadmap/evidence/dataset_validation

# Runtime task audit and sanity floors
python scripts/audit_tasks.py --out docs/roadmap/evidence/task_audit.json
python scripts/sanity_baselines.py --shard 0/8 --out part_0.csv --episodes-out episodes_0.csv
python scripts/aggregate_sanity.py --dir <folder with part_*.csv and episodes_*.csv>
```

Episode seeds are `42 + 1000 * task_number + k` for generation and `200000 + 1000 * task_number + j` for evaluation.

### Dataset schema (Robomimic HDF5)

`data/demo_i/{actions (T,7) float32, rewards (T,), dones (T,), states (T,85) float64, obs/*, next_obs/*}` with
`obs/agentic_rgb (T,640,660,3)`, `obs/wrist_rgb (T,300,420,3)`, `obs/{joint_positions, joint_velocities, tcp_pose,
gripper_width}` and robomimic/LIBERO aliases (hard links); per-demo `seed`, `init_state`, `language_instruction`;
root `env_args`, `problem_info`, `model_file`, camera intrinsics and poses; `mask/{train, valid}` (45/5).

Known limitations of the released files (details in `docs/roadmap/AUDIT_REPORT.md`):
- no provenance attributes (code version, simulator version, command); produced by at least three generator runs;
- the state after the last action is not stored and `next_obs[T-1]` repeats `obs[T-1]`;
- 114 demos exceed the 500-step environment limit (130 exceed the evaluation budget H = 400);
- per-demo `language_instruction` rotates over four variants; the metric variant states nominal, not actual, start
  coordinates;
- recorded actions are float32 roundings of the executed float64 actions, so replaying them reproduces the success
  step for 595 of 630 sampled demos (median deviation 0.02 mm) rather than all;
- 6 files store observation aliases as copies (larger files, same content).

Files generated with the current generator carry provenance, the final observation and state, and canonical
instructions, contain only demos within the environment limit, and replay bit-identically. See also
`docs/DATASET_CARD.md`.

## Training and evaluation

```bash
# Low-dim (proprioception + object pose), image + proprioception, or proprioception-only (blind ablation)
python scripts/train_policy.py --policy bc --modality lowdim --domain tabletop --tier core --epochs 200 --seed 1
python scripts/evaluate_policy_suite.py --checkpoint checkpoints/bc_lowdim_tabletop_core_seed1_best.pt \
    --domain tabletop --tier core --suites nominal,p2_spatial,p3_visual,p4_physical,combined_ood --episodes 50 \
    --out-dir results/bc_lowdim_tabletop_seed1
python scripts/evaluate_policy_suite.py --policy oracle --task tabletop_sort_red --episodes 10
```

`evaluate_policy_suite.py` writes one row per episode (seed, success, steps, peak contact force) and a summary with
Wilson and task-clustered bootstrap intervals, PRI and GDI against nominal on paired seeds, and the force violation
rate (F_max = 150 N; the oracle itself exceeds it in 4.8% of episodes). `evaluate_id.py` and `evaluate_ood.py` are thin
wrappers around it. Checkpoints store their input normalizer, the dataset files, and the code provenance, and are
selected by validation loss.

## Repository layout

```text
afford_pro/            package: task definitions and tiers, environments, oracles, perturbations,
                       datasets (writer, generation, validation, replay, converters), evaluation, learning, provenance
scripts/               generation, validation, audits, sanity baselines, sweeps, training, evaluation, converters
tests/                 unit tests (registry, validation, generation, recording, converters, prompts, frontier)
configs/               multi-tier instruction variants with grounding flags
docs/roadmap/          plan, progress, audit, claim ledger, experiment log, literature comparison, evidence/
docs/quarantine/       withdrawn synthetic projections (not results)
AFFORD-Bench_RSS'27/   paper source (main.tex, references.bib, figures/)
```

## Citation

```bibtex
@inproceedings{affordbench2027,
  title     = {AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning},
  author    = {Anonymous},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2027},
  note      = {Under review}
}
```
