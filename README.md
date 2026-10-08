# AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning

[![Target Venue: RSS 2027](https://img.shields.io/badge/Submission-RSS%202027%20Stage%201-blue.svg)](AFFORD-Bench_RSS'27/paper-template-latex/main.pdf)
[![Physics: MuJoCo Native](https://img.shields.io/badge/MuJoCo-3.x%20C%2B%2B%20Native-brightgreen.svg)](https://mujoco.org)
[![Dataset Standard: Robomimic HDF5](https://img.shields.io/badge/Dataset-Robomimic%20HDF5-orange.svg)](https://robomimic.github.io)
[![Scale: 210 Tasks / 10,500 Demos Target](https://img.shields.io/badge/Scale-210%20Tasks%20%7C%2010%2C500%20Demos%20Target-purple.svg)]()
[![Suites: 35 Benchmark Suites](https://img.shields.io/badge/Suites-35%20Benchmark%20Suites-darkblue.svg)]()
[![Zero External Cost](https://img.shields.io/badge/External%20Cost-%240.00%20Zero-success.svg)]()

**AFFORD-Bench** is a large-scale, **policy-agnostic**, affordance-grounded robot manipulation benchmark and dataset suite engineered to stress-test the out-of-distribution (OOD) physical generalization of any robot learning policy---ranging from **Classical Imitation Learning (BC, BeT)** and **Visuomotor Action Chunking (Diffusion Policy, ACT)** to **3D Point-Cloud Policies (DP3)**, **Vision-Language-Action (VLA) Foundation Models (OpenVLA, SmolVLA, AtomVLA, Octo)**, and **Offline Reinforcement Learning (CQL, IQL)**.

Operating on a 7-DoF Franka Emika Panda manipulator simulated with contact physics in MuJoCo, AFFORD-Bench provides **210 formally defined physical manipulation tasks** across **7 distinct domains**, accompanied by a 100% local deterministic demonstration generator targeting **up to 10,500 demonstration episodes** (>2.2M physical timesteps), synchronized dual-camera perception ($640\times 660$ workcell and $300\times 420$ wrist), and optional **true physical 3D metric depth** (`--depth`).

> 🌐 **Interactive Project Page**: Explore multi-modal videos, interactive domain tabs, perturbation hierarchies, and dataset schemas in [`docs/index.html`](docs/index.html).

---

## 1. Overview: Physical Affordances & Policy-Agnostic Benchmark Architecture

Recent investigations by **LIBERO** (130 tasks, 50 core), **LIBERO-PRO** (40 evaluation tasks, 4 perturbation axes), and **LIBERO-X** (600 tasks across Kitchen, Study, and Living scenes, shipping 2,520 VR teleoperated demonstrations) revealed that seemingly high success rates on existing imitation learning benchmarks often reflect rote memorization of nominal camera views and initial poses rather than generalized physical understanding. When exposed to minor spatial displacements or visual shifts, policy performance degrades precipitously.

AFFORD-Bench grounds robot learning in **ecological contact mechanics** across a **policy-agnostic** architecture:
- **Universal Policy Support**: Directly ingestible by BC, ACT, Diffusion Policy, 3D Point-Cloud policies (DP3), VLAs (OpenVLA, SmolVLA, AtomVLA), and Offline RL (CQL, IQL).
- **Coulomb Friction Variation ($\mu \in [0.10, 1.20]$)**: Grasping forces must prevent slippage without crushing objects.
- **Dynamic Payload Scaling ($0.5\times$--$3.0\times$)**: Transport dynamics require inertial compensation.
- **Sub-Millimeter Clearances ($\pm 1.5\,\text{mm} \le \epsilon \le \pm 20.0\,\text{mm}$)**: Assembly requires compliance along insertion manifolds.
- **Multi-Tier Language Grounding**: Multi-tier linguistic descriptions per task (colloquial, geometric, physical affordance, atomic subtasks in `configs/prompts_augmented.yaml`) evaluating physical placement tolerances.
- **Synchronized Multi-Modal Perception**: Dual-view RGB ($640\times 660$ and $300\times 420$) aligned with optional metric depth ($D_{\text{3D}}$ via `--depth`).

### Benchmark Positioning: LIBERO, LIBERO-PRO, LIBERO-X, and AFFORD-Bench
- **LIBERO** (arXiv 2306.03310): Released dataset of 6,500 human SpaceMouse teleoperated demos across 130 tasks (50 core), evaluated with BDDL predicate conjunctions.
- **LIBERO-PRO** (arXiv 2510.03827): Evaluation-only protocol evaluating 40 tasks across 4 perturbation axes (object, spatial, semantic, environment).
- **LIBERO-X** (arXiv 2602.06556): Procedural benchmark with 600 tasks across diverse scenes, 2,520 VR teleoperated training demonstrations, and a 5-level hierarchical evaluation protocol (L1--L5).
- **AFFORD-Bench**: 210 formally specified manipulation tasks across 7 domains $\times$ 5 hierarchical perturbation regimes (Nominal, Spatial, Physical-Coulomb friction & mass scaling, Visual, Composite OOD) = **35 Comprehensive Benchmark Suites**. Crucially, unlike LIBERO suites, AFFORD-Bench checks success physically (released, at rest, supported, within 2–8 mm tolerance tiers with zero artificial welds or magnetic snapping) and perturbs physical contact dynamics ($\mu \in [0.10, 1.20]$, payload mass $0.5\times$--$3.0\times$).

---

## 2. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/s-elim/AFFORD_Bench.git
cd AFFORD_Bench

# Create and activate conda environment
conda create -n afford_bench python=3.10 -y
conda activate afford_bench

# Install package in editable mode with dependencies
pip install -e .
```

---

## 3. Demonstration Data Generation & Integrity Verification

AFFORD-Bench provides local, deterministic demonstration synthesis powered by closed-loop kinematic oracles ($0.00 external API cost):

```bash
# Generate 50 demonstrations per task for a specific domain (optionally append --depth for metric depth)
python scripts/generate_all_150_tasks.py --domain industrial --num-demos 50
python scripts/generate_all_150_tasks.py --domain kitchen --num-demos 50
python scripts/generate_all_150_tasks.py --domain tabletop --num-demos 50
python scripts/generate_all_150_tasks.py --domain lab --num-demos 50
python scripts/generate_all_150_tasks.py --domain long_horizon --num-demos 50
python scripts/generate_all_150_tasks.py --domain chain_precision --num-demos 50
python scripts/generate_all_150_tasks.py --domain chain_service --num-demos 50

# Or generate all 210 tasks across all domains with 3D metric depth
python scripts/generate_all_150_tasks.py --domain all --num-demos 50 --depth

# Export 50 pre-settled initial simulator states per task (for zero-variance benchmark evaluation)
python scripts/export_initial_states.py --domain all

# Verify dataset integrity (shapes, channels, metric depth, robomimic aliases, and step counts)
python scripts/verify_dataset_integrity.py
```

---

## 4. Policy Training (`scripts/train_policy.py`)

Train robot manipulation policies (Diffusion Policy and Behavioral Cloning) on the generated Robomimic HDF5 demonstration datasets. Supports both ultra-fast state-based training and full visuomotor training.

### Fast State-Based Policy Training
Trains on 22-dimensional proprioceptive states (joint positions, joint velocities, end-effector Cartesian pose, and gripper width) and 7D relative action chunks:

```bash
# Train Diffusion Policy on Industrial domain (runs in minutes on CPU/GPU)
python scripts/train_policy.py --policy diffusion --modality state --domain industrial --epochs 20 --batch-size 64

# Train Behavioral Cloning (BC) policy on Kitchen domain
python scripts/train_policy.py --policy bc --modality state --domain kitchen --epochs 25 --batch-size 64

# Train Diffusion Policy on Multi-Stage Precision Chaining domain
python scripts/train_policy.py --policy diffusion --modality state --domain chain_precision --epochs 25 --batch-size 64

# Train on a single specific task
python scripts/train_policy.py --policy diffusion --modality state --task industrial_shaft_assembly --epochs 20
```

### Visuomotor Policy Training
Trains end-to-end visuomotor representations consuming agentic camera observations ($640\times 660$), wrist observations ($300\times 420$), and proprioception:

```bash
# Train Visuomotor Diffusion Policy on Tabletop domain
python scripts/train_policy.py --policy diffusion --modality visual --domain tabletop --epochs 30 --batch-size 32

# Train Visuomotor BC Policy on Cleanroom Laboratory domain
python scripts/train_policy.py --policy bc --modality visual --domain lab --epochs 30 --batch-size 32
```

**Training CLI Options**:
- `--policy`: Policy architecture (`diffusion` or `bc`).
- `--modality`: Observation input modality (`state` or `visual`).
- `--domain`: Target domain (`industrial`, `kitchen`, `tabletop`, `lab`, `long_horizon`, `chain_precision`, `chain_service`, or `all`).
- `--task`: Optional single task ID (e.g. `industrial_shaft_assembly`).
- `--epochs`: Total training epochs (default: `20`).
- `--batch-size`: Batch size (default: `64`).
- `--chunk-size`: Action chunking horizon $H$ (default: `16`).
- `--save-dir`: Checkpoint output directory (default: `checkpoints/`).

---

## 5. In-Distribution (ID) Policy Evaluation (`scripts/evaluate_id.py`)

Evaluate trained checkpoints or reference baselines under standard **In-Distribution (Nominal)** conditions using disjoint test seeds:

```bash
# Evaluate a trained policy checkpoint on an entire domain
python scripts/evaluate_id.py --checkpoint checkpoints/diffusion_state_industrial_best.pt --domain industrial --num-episodes 20

# Evaluate a trained checkpoint on a single task
python scripts/evaluate_id.py --checkpoint checkpoints/diffusion_state_industrial_shaft_assembly_best.pt --task industrial_shaft_assembly --num-episodes 20

# Evaluate reference closed-loop Oracle baseline
python scripts/evaluate_id.py --policy oracle --domain kitchen --num-episodes 10

# Evaluate Random exploration baseline
python scripts/evaluate_id.py --policy random --domain tabletop --num-episodes 5
```

**Output Metrics**:
- **Success Rate ($SR_{\text{ID}}$)**: Percentage of successful task completions under nominal conditions.
- **Average Episode Length**: Mean physical execution steps to task success.
- **Execution Throughput**: Step execution latency and real-time factor.

---

## 6. Out-Of-Distribution (OOD) Policy Evaluation (`scripts/evaluate_ood.py`)

Benchmark policy resilience against environmental shifts across the **5 canonical perturbation regimes**:

1. **Spatial-Kinematic (`p2_spatial`)**: Coordinate offsets $\pm 6\,\text{cm}$, yaw rotations $\pm 30^\circ$, tolerance contraction.
2. **Physical-Dynamic (`p1_object` / `p4_physical`)**: Coulomb friction $\mu \in [0.10, 1.20]$, payload mass scaling $0.5\times$--$3.0\times$.
3. **Visual-Photometric (`p3_visual`)**: Illumination shifts $[0.4, 2.2]\,I_0$, randomized lighting direction, camera jitter.
4. **Composite Multi-Axis OOD (`combined_ood`)**: Simultaneous multi-axis distribution shift.
5. **All Regimes (`all`)**: Evaluates across all five regimes and computes aggregate robustness indices.

```bash
# Evaluate a trained policy checkpoint across all OOD regimes on Industrial domain
python scripts/evaluate_ood.py --checkpoint checkpoints/diffusion_state_industrial_best.pt --domain industrial --perturbation all --num-episodes 10

# Evaluate reference Oracle across Physical perturbations on Kitchen domain
python scripts/evaluate_ood.py --policy oracle --domain kitchen --perturbation p1_object --num-episodes 10

# Evaluate against Composite Multi-Axis OOD shifts
python scripts/evaluate_ood.py --policy oracle --domain tabletop --perturbation combined_ood --num-episodes 10
```

**Quantitative Generalization Metrics Computed**:
- **Perturbation Resilience Index (PRI)**:
  $$\text{PRI}(\mathcal{P}) = \frac{\text{SR}(\mathcal{P})}{\text{SR}(\text{Nominal})} \in [0.0, 1.0] \quad (\text{Higher is better})$$
- **Generalization Degradation Index (GDI)**:
  $$\text{GDI}(\mathcal{P}) = \frac{\text{SR}(\text{Nominal}) - \text{SR}(\mathcal{P})}{\text{SR}(\text{Nominal})} \times 100\% \quad (\text{Lower is better})$$

---

## 7. Generating OOD Evaluation Suites (`scripts/generate_ood_eval_suites.py`)

Generate offline perturbed evaluation demonstration suites for benchmark distribution shift testing:

```bash
# Generate Spatial OOD evaluation suite for Industrial domain
python scripts/generate_ood_eval_suites.py --domain industrial --perturbation p2_spatial --num-demos 10

# Generate Physical-Dynamic OOD suite for Kitchen domain
python scripts/generate_ood_eval_suites.py --domain kitchen --perturbation p1_object --num-demos 10

# Generate Composite Multi-Axis OOD evaluation suites across all domains
python scripts/generate_ood_eval_suites.py --domain all --perturbation combined_ood --num-demos 5
```

---

## 8. Benchmark Taxonomy (7 Domains $\times$ 30 Tasks = 210 Tasks)

| Domain | Task Numbers | Tasks | Dominant Physics & Affordances | Clearances |
| :--- | :---: | :---: | :--- | :---: |
| **D1: Industrial Precision** | 001–030 | 30 | High-stiffness assembly, splined shafts, gear meshing, toggle clamping | $\pm 2.0\text{--}5.0\,\text{mm}$ |
| **D2: Culinary Operations** | 031–060 | 30 | Spatula burger patty flipping, portafilter espresso docking, food placement | $\pm 5.0\text{--}12.0\,\text{mm}$ |
| **D3: Domestic Tabletop** | 061–090 | 30 | Color block sorting, drawer slide actuation, caddy packing, cup stacking | $\pm 8.0\text{--}20.0\,\text{mm}$ |
| **D4: Cleanroom Laboratory** | 091–120 | 30 | Micropipette tip pickup, centrifuge vial docking, reagent mixing, tube transfer | $\pm 1.5\text{--}4.0\,\text{mm}$ |
| **D5: Long-Horizon Sequences**| 121–150 | 30 | Multi-stage assembly, sequential tool acquisition, unlatching, sub-assembly chaining | $\pm 3.0\text{--}10.0\,\text{mm}$ |
| **D6: Precision Chaining** | 151–180 | 30 | Chained 2–3 step industrial operations: shaft-bearing-clamp, gasket-bolt-torque | $\pm 1.8\text{--}4.5\,\text{mm}$ |
| **D7: Service Assembly** | 181–210 | 30 | Chained 2–3 step culinary service: patty-cheese-bun, mug-brew-serve, plate-fork | $\pm 4.0\text{--}10.0\,\text{mm}$ |
| **Total Benchmark Scope** | **001–210** | **210** | **7 Domains $\times$ 30 Tasks = 210 Tasks (35 Evaluation Suites)** | $\pm 1.5\text{--}20.0\,\text{mm}$ |

---

## 9. Repository Layout

```text
AFFORD_Bench/
├── README.md                           # Master documentation & benchmark usage
├── setup.py                            # Pip installable package (afford_pro)
├── afford_pro/
│   ├── __init__.py                     # API: make(), list_tasks(), get_oracle()
│   ├── registry.py                     # Master 210-task & oracle registry
│   ├── task_definitions.py            # Formal definitions for all 210 manipulation tasks
│   ├── envs/                           # Gymnasium environments
│   │   ├── base_franka_env.py          # Franka Panda MuJoCo base env with 3D metric depth
│   │   ├── industrial/                 # Domain 1: Industrial Precision Assembly (Tasks 001–030)
│   │   ├── kitchen/                    # Domain 2: Culinary Operations (Tasks 031–060)
│   │   ├── tabletop/                   # Domain 3: Domestic Manipulation (Tasks 061–090)
│   │   ├── lab/                        # Domain 4: Laboratory Sciences (Tasks 091–120)
│   │   └── long_horizon/               # Domain 5, 6, 7: Long-Horizon & Chained Sequences (Tasks 121–210)
│   ├── oracles/                        # Deterministic Closed-Loop Algorithmic Solvers
│   ├── perturbations/                  # Affordance-Grounded Perturbation Engine (Spatial, Physics, Visual)
│   └── datasets/                       # Universal Robomimic HDF5 dataset writer
├── AFFORD-Bench_RSS'27/
│   └── paper-template-latex/           # RSS 2027 Stage 1 Extended Abstract LaTeX source
│       ├── main.tex                    # Strict 6-page manuscript source
│       ├── main.pdf                    # Compiled publication-grade PDF
│       ├── references.bib              # Verified bibliography
│       └── figures/                    # PDF and PNG publication figures
└── scripts/
    ├── train_policy.py                 # Ready-to-run PyTorch policy training harness
    ├── evaluate_id.py                  # In-Distribution (ID) evaluation script
    ├── evaluate_ood.py                 # Out-Of-Distribution (OOD) evaluation script
    ├── generate_ood_eval_suites.py     # OOD evaluation suite dataset generator
    ├── generate_all_150_tasks.py       # Batch demonstration synthesis pipeline
    ├── verify_dataset_integrity.py     # Dataset schema and metric depth verification script
    └── generate_paper_figures.py       # High-resolution multi-modal visualization generator
```

---

## 10. Citation

If you use **AFFORD-Bench** or its datasets in your research, please cite our RSS 2027 Stage 1 Extended Abstract:

```bibtex
@inproceedings{affordbench2027,
  title     = {AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning},
  author    = {Anonymous},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2027},
  note      = {Stage 1 Extended Abstract}
}
```
