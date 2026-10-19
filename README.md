# AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning

AFFORD-Bench is a MuJoCo benchmark for robot manipulation policies on a simulated Franka Emika Panda: 210 tasks in 7
domains over six workcells, from shaft and pin seating to burger assembly, espresso service, and centrifuge loading.
Every instruction names only simulated objects and mechanisms and is checked as a conjunction of physical goals, in
order. A placed object must be released, supported, upright, and at rest within 2 mm and 5 deg of its goal in the
precision domains (8 mm and 15 deg elsewhere). Grasping is frictional; there is no attachment, snapping, or teleport.
Demonstrations come from a privileged closed-loop oracle and are stored as Robomimic HDF5 with converters to LeRobot
and RLDS.

Project page: `docs/index.html` (GitHub Pages).

```python
import afford_pro
env = afford_pro.make("long_horizon_espresso_service")   # "Put the mug on the drip tray, press the brew button,
oracle = afford_pro.get_oracle("long_horizon_espresso_service", env)  #  then move the mug to the serving tray."
obs, info = env.reset(seed=0)
```

## Installation

```bash
conda create -n afford_bench python=3.10 -y && conda activate afford_bench
pip install -e .            # pins mujoco==3.5.0, the build used for all data and numbers
pip install torch           # only for training/evaluating learned policies
python -m pytest -q tests   # about 2,400 tests, a few minutes on CPU
```

All reported numbers use MuJoCo 3.5.0 and the Panda meshes in `afford_pro/assets/robots/assets/`. Results with other
MuJoCo builds can differ, because contact resolution changes between releases.

## Tasks

| Domain | Tasks | Workcells | Goals per task | Tolerance (xy, tilt) |
| --- | --- | --- | --- | --- |
| D1 Industrial precision | 001-030 | assembly jig, electronics bench | 1 | 2 mm, 5 deg |
| D2 Culinary operations | 031-060 | burger station, espresso station | 1 | 8 mm, 15 deg |
| D3 Domestic tabletop | 061-090 | sorting table | 1-2 | 8 mm, 15 deg |
| D4 Laboratory | 091-120 | laboratory bench | 1 | 2 mm, 5 deg |
| D5 Long-horizon | 121-150 | all six | 2-5 | workcell's |
| D6 Precision chains | 151-180 | assembly jig, electronics, laboratory | 2-4 | 2 mm, 5 deg |
| D7 Service chains | 181-210 | burger, espresso, sorting | 2-6 | 8 mm, 15 deg |

Goal predicates (`afford_pro/bench/spec.py`, checked in `afford_pro/bench/predicates.py`):

- `On(body, target)`: released, supported, at rest, upright, lifted at least 2 cm, and at its goal pose on a point
  (a bore, slot, rack hole), inside a region (a bin, compartment, marked square), or on another body (stacking).
- `Pressed(button)` (latching buttons), `Opened`/`Closed(slider)` (drawer, vise jaw), `Slid(body, target)` (wiping),
  `HeldOver(body, target, clearance)`, `GripperAbove(target, height)`, `Touching(body)`, `Lifted(body, height)`.
- Goals may be ordered: a goal counts only when it becomes true after the goals before it; a waypoint goal (`once`)
  must be reached but need not hold at the end. The step budget is 400 steps plus 250 per further goal.

Every goal pose is found by letting the body come to rest in its target when the task is built, so every goal is
attainable. The full table of tasks, goals, starts and budgets is `docs/roadmap/T9_TASK_TABLE.md`.

## Validation

`scripts/validate_bench_tasks.py` runs every task against the acceptance gate (seeds disjoint from evaluation and data
generation; evidence in `docs/roadmap/evidence/task_validation_*.json`):

| Check | Result |
| --- | --- |
| Every goal physically attainable | checked when each task is built |
| Scripted oracle, nominal | 2096/2100 episodes (99.8%), every task >= 9/10 |
| Random / null policies | 0/420 / 0/420 |
| Starts (4200 resets) | no goal holds at reset; 225 region starts with s.d. >= 0.99 cm; 31 pre-placed assemblies; 22 two-slot starts |
| Oracle under L4 / L2 / L5 | 98.9% / 97.5% / 95.5% (the ceiling of each suite) |

## Perturbation suites (`afford_pro/perturbations/engine.py`)

| Suite | What changes |
| --- | --- |
| `nominal` (L1) | Each moved object starts uniformly in its own region (half-widths 2-40 mm, yaw up to +-45 deg), in one of two receptacles, or pre-assembled; TCP jitter +-1.5 cm. Rejection-sampled (separation, no fixture overhead, a collision-free grasp, no goal already holding) |
| `p1_object` | Color of the moved objects x U(0.75, 1.25) |
| `p2_spatial` (L2) | Region starts offset by U(7, 12) cm in a random direction, yaw magnitude U(45, 90) deg |
| `p3_visual` (L3) | Light position +-0.5 m, diffuse x U(0.5, 1.5) clipped at 1 |
| `p4_physical` (L4) | Sliding friction U(0.12, 0.25) on the moved objects and finger pads; their mass and inertia x U(1.8, 2.5) |
| `combined_ood` (L5) | L2 + L3 + L4 + color |

Suites are paired: an evaluation seed gives the same layout in L1, L3, L4 and color, and the same layout in L2 and L5.

## Data generation and validation

```bash
# Generate (skips valid files, refuses to overwrite invalid ones or files of another task set, records provenance,
# writes atomically)
python scripts/generate_all_150_tasks.py --num-demos 50
python scripts/generate_demos.py --task tabletop_sort_red --num-demos 5 --output-dir datasets/scratch

# Validate every episode (+ replay 3 demos per file; replays must reproduce every recorded state exactly)
python scripts/validate_datasets.py --replay-per-file 3 --workers 8 --out-dir results/validation
```

Episode seeds are `42 + 1000 * task_number + k` for generation and `200000 + 1000 * task_number + j` for evaluation.
Schema and guarantees: `docs/DATASET_CARD.md`.

## Training and evaluation

```bash
# Image + proprioception, privileged object poses + proprioception, or proprioception only (blind ablation)
python scripts/train_policy.py --policy bc --modality objects --domain tabletop --epochs 200 --seed 1 --tag seed1
python scripts/evaluate_policy_suite.py --checkpoint checkpoints/bc_objects_tabletop_seed1_best.pt --domain tabletop \
    --suites nominal,p1_object,p2_spatial,p3_visual,p4_physical,combined_ood --episodes 50 \
    --out-dir results/bc_objects_tabletop_seed1
python scripts/evaluate_policy_suite.py --policy oracle --task tabletop_sort_red --episodes 10
```

`evaluate_policy_suite.py` writes one row per episode (seed, success, steps, peak contact force) and a summary with
Wilson and task-clustered bootstrap intervals, PRI and GDI against nominal on paired seeds, and the force violation
rate (F_max = 150 N). Success counts within each task's step budget. Checkpoints store their input normalizer, the
dataset files, and the code provenance. GPU baselines: `docs/BASELINE_RUNBOOK.md`.

## Repository layout

```text
afford_pro/bench/      task specifications, goal predicates, multi-goal environment, oracle, language
afford_pro/            registry, base environments, perturbations, datasets (writer, generation, validation,
                       replay, converters), evaluation, learning, provenance
scripts/               scene build, task validation, generation, dataset validation, training, evaluation, page data
tests/                 unit tests
docs/                  project page, dataset card, baseline runbook, roadmap and evidence
```

The earlier single-body task set is deprecated; its code remains reachable with
`afford_pro.make(task_id, task_set="legacy")` only to read and replay files written for it.

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
