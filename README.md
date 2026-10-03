# AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning

[![Target Venue: RSS 2027](https://img.shields.io/badge/Submission-RSS%202027%20Stage%201-blue.svg)](AFFORD-PRO_RSS'27/paper-template-latex/main.pdf)
[![Physics: MuJoCo Native](https://img.shields.io/badge/MuJoCo-3.x%20C%2B%2B%20Native-brightgreen.svg)](https://mujoco.org)
[![Dataset Standard: Robomimic HDF5](https://img.shields.io/badge/Dataset-Robomimic%20HDF5-orange.svg)](https://robomimic.github.io)
[![Scale: 150 Tasks / 7,500 Demos](https://img.shields.io/badge/Scale-150%20Tasks%20%7C%207%2C500%20Demos-purple.svg)]()
[![Zero External Cost](https://img.shields.io/badge/External%20Cost-%240.00%20Zero-success.svg)]()

**AFFORD-Bench** is a large-scale, affordance-grounded robot manipulation benchmark and dataset suite engineered to evaluate and stress-test the out-of-distribution (OOD) generalization of **Vision-Language-Action (VLA) foundation models** (OpenVLA, SmolVLA, AtomVLA, Octo, $\pi_0$) and **visuomotor policy learning representations** (Diffusion Policy, ACT, DP3).

Operating on a 7-DoF Franka Emika Panda manipulator simulated in high-fidelity MuJoCo contact physics, AFFORD-Bench features **150 formally defined physical tasks** across **5 diverse manipulation domains**, backed by **7,500 expert demonstrations** (>1.5M physical timesteps), synchronized dual-view RGB perception ($640\times 660$ workcell and $300\times 420$ wrist), and **true physical 3D metric depth** ($D_{3D}$).

---

## 1. Motivation: Beyond Appearance Memorization to Physical Affordance Grounding

Recent post-hoc evaluations of imitation learning suites, such as **LIBERO-PRO** and **LIBERO-X**, exposed a critical vulnerability in current robot learning benchmarks: policies achieving $>90\%$ success under nominal training layouts suffer catastrophic failure (collapsing to $<15\%$) under minor spatial offsets or camera jitter. Existing benchmarks attribute this solely to visual distribution shifts, evaluating models primarily in narrow domestic toy scenes.

However, real-world robotic manipulation is fundamentally governed by **physical affordances**---contact mechanics, surface friction, kinematic compliance, mass distributions, and sub-millimeter clearance tolerances:
- An insertion policy must align along the assembly manifold without kinematic jamming.
- A grasping policy must adapt normal forces to varying Coulomb friction ($\mu$) without slippage.
- A tool-use policy must chain multi-stage kinematic pre-conditions without compounding error.

| Dimension | Standard Benchmarks (LIBERO, RLBench) | LIBERO-PRO / LIBERO-X | **AFFORD-Bench (Ours)** |
| :--- | :--- | :--- | :--- |
| **Task Count** | 10–130 tasks | 10–40 tasks (tabletop subsets) | **150 Unique Physical Tasks** (30 per domain) |
| **Domain Diversity** | Single domestic tabletop | Single domestic tabletop | **5 Real-World Domains** (Industrial, Kitchen, Domestic, Cleanroom, Long-Horizon) |
| **Visual Modalities** | Single 2D camera ($128\times 128$) | Single 2D camera ($128\times 128$) | **Dual High-Res RGB** ($640\times 660$ workcell + $300\times 420$ wrist) |
| **3D Perception** | No depth (`depth: []`) | No depth | **True 3D Metric Depth** ($D_{3D}$) in meters |
| **Contact Dynamics** | Static contact parameters | Static friction/mass | **Dynamic Coulomb Friction** ($\mu \in [0.1, 1.2]$) & **Mass Scaling** ($0.5\times$--$3.0\times$) |
| **Kinematic Precision**| Loose clearances ($>20\,\text{mm}$) | Coarse tolerances | **Precision Clearances** ($\pm 1.5\,\text{mm} \le \epsilon \le \pm 20.0\,\text{mm}$) |
| **Synthesis Pipeline**| Human teleoperation (expensive) | Manual demonstration replay | **Analytical Closed-Loop Kinematic Oracles** (40–50 FPS, $0.00 external cost) |
| **Benchmark Standard**| Custom formats | Custom evaluation scripts | **Standard Robomimic HDF5** out-of-the-box compatibility |

---

## 2. The 150-Task Domain Taxonomy (5 Domains $\times$ 30 Tasks)

AFFORD-Bench is partitioned into five specialized manipulation domains, each containing exactly 30 distinct tasks:

| Domain | Task IDs | Count | Representative Manipulation Tasks | Clearance ($\epsilon$) | Dominant Contact Physics |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **D1: Industrial Precision** | 001–030 | 30 | Gear meshing, splined shaft seating, flange bolt insertion, fixture toggle clamping | $\pm 2.0\text{--}5.0\,\text{mm}$ | High stiffness, rigid peg-in-hole friction |
| **D2: Culinary Operations** | 031–060 | 30 | Spatula burger patty flipping, espresso portafilter docking, cup positioning, slicing | $\pm 5.0\text{--}12.0\,\text{mm}$ | Deformable contact, non-planar slipping |
| **D3: Domestic Tabletop** | 061–090 | 30 | Color block sorting, drawer slide actuation, caddy packing, plate stacking | $\pm 8.0\text{--}20.0\,\text{mm}$ | Multi-body clutter, dynamic toppling |
| **D4: Laboratory Sciences** | 091–120 | 30 | Micropipette tip pickup, centrifuge vial docking, reagent mixing, tube rack transfer | $\pm 1.5\text{--}4.0\,\text{mm}$ | Precision alignment, fragile glass contact |
| **D5: Long-Horizon Sequences**| 121–150 | 30 | Multi-stage assembly, sequential tool acquisition, unlatching, sub-assembly chaining | $\pm 3.0\text{--}10.0\,\text{mm}$ | Cumulative error accumulation, sub-goal chaining |

---

## 3. The Affordance-Grounded Perturbation Manifold ($\mathcal{P}$)

To expose whether policy representations encode physical affordances or merely memorize visual landmarks, AFFORD-Bench defines a structured perturbation space $\mathcal{P} = \mathcal{P}_{\mathrm{spatial}} \times \mathcal{P}_{\mathrm{physics}} \times \mathcal{P}_{\mathrm{visual}}$:

1. **Spatial-Kinematic Shifts ($\mathcal{P}_{\mathrm{spatial}}$)**:
   - Position translations: $\mathbf{p}_{\text{init}} \sim \mathcal{U}(\mathbf{p}_0 - \boldsymbol{\delta}_{\mathbf{p}}, \mathbf{p}_0 + \boldsymbol{\delta}_{\mathbf{p}})$ with $\boldsymbol{\delta}_{\mathbf{p}} = [\pm 6\,\text{cm}, \pm 6\,\text{cm}, 0\,\text{cm}]^T$.
   - Yaw rotations: $\theta_{\text{yaw}} \sim \mathcal{U}(-30^\circ, +30^\circ)$.
   - Clearance contraction: insertion tolerances systematically contracted from nominal $\epsilon_0 = 5.0\,\text{mm}$ down to $\epsilon_{\text{tight}} = 1.5\,\text{mm}$.
2. **Physical-Dynamic Shifts ($\mathcal{P}_{\mathrm{physics}}$)**:
   - Coulomb friction coefficient: $\mu_{\text{contact}} \sim \mathcal{U}(0.10, 1.20)$ (nominal $\mu_0 = 0.70$).
   - Object payload mass: $m_{\text{obj}} \sim \mathcal{U}(0.5\,m_0, 3.0\,m_0)$.
   - Dynamic contact damping: randomized restitution and rotational friction.
3. **Visual-Photometric Shifts ($\mathcal{P}_{\mathrm{visual}}$)**:
   - Illumination intensity: $I_{\text{light}} \in [0.4, 2.2]\,I_0$.
   - Directional light vectors: randomized over hemisphere $\mathbf{v}_{\text{light}} \in \mathbb{S}^2$.
   - Camera jitter: extrinsic rotational perturbation $\pm 2^\circ$.
4. **Composite OOD Shift ($\mathcal{P}_{\mathrm{comp}}$)**:
   - Joint simultaneous perturbation across all three axes, testing holistic out-of-distribution resilience.

---

## 4. Training Protocols & Implementation Details

### Observation & Action Parameterization
- **Observation Space**: $\mathbf{o}_t = (\mathbf{I}_{\text{work}}, \mathbf{I}_{\text{wrist}}, \mathbf{D}_{\text{3D}}, \mathbf{q}_t)$
  - Workcell RGB: $\mathbf{I}_{\text{work}} \in \mathbb{R}^{224\times 224\times 3}$ (downsampled from $640\times 660$)
  - Wrist RGB: $\mathbf{I}_{\text{wrist}} \in \mathbb{R}^{224\times 224\times 3}$ (downsampled from $300\times 420$)
  - 3D Metric Depth: $\mathbf{D}_{\text{3D}} \in \mathbb{R}^{224\times 224\times 1}$ (float32 physical depth in meters)
  - Proprioceptive State: $\mathbf{q}_t \in \mathbb{R}^{21}$ (joint positions, joint velocities, EEF pose, gripper displacement)
- **Action Space**: Relative Cartesian end-effector velocity $\mathbf{a}_t = [\Delta \mathbf{p}_t, \Delta \boldsymbol{\theta}_t, a_{\text{grip}, t}] \in \mathbb{R}^7$, normalized to $[-1, 1]$ via demonstration statistics:
  - Translation: $\Delta \mathbf{p}_t \in [-0.05, 0.05]\,\text{m}$
  - Rotation: Axis-angle $\Delta \boldsymbol{\theta}_t$
  - Gripper: $a_{\text{grip}, t} \in \{-1, +1\}$

### Benchmarked Policy Architectures
- **Diffusion Policy (DP)**: 1D Temporal U-Net with DDPM (100 diffusion steps, square-cosine schedule), action horizon $H=16$, $T_{\text{obs}}=2$. Optimized via AdamW ($\text{lr}=10^{-4}$, weight decay $10^{-4}$) for 100k gradient steps per domain at batch size 64.
- **Action Chunking with Transformers (ACT)**: C-VAE with 8-layer Transformer encoder-decoder predicting chunks $H=16$, trained using L1 reconstruction and KL regularization ($\beta=10$) for 100k steps.
- **OpenVLA (7B)**: Llama-2-7B language backbone with SigLIP-SO400M vision encoder, fine-tuned via LoRA ($r=32, \alpha=64$) on attention projections with prompt template *"What action should the robot take to [Task Instruction]?"* and 256-bin action discretization, optimized via AdamW ($\text{lr}=5\times 10^{-5}$) for 30k steps per domain.
- **SmolVLA (450M)**: Compact edge VLA based on SmolVLM-2 with a flow-matching action expert head, trained end-to-end with continuous flow-matching objectives for 50k steps.
- **AtomVLA**: Subtask-aware post-training incorporating LLM atomic decomposition and predictive latent world model rollouts.
- **3D Diffusion Policy (DP3)**: Point-cloud-conditioned diffusion policy consuming $N_p=2048$ points extracted from metric depth $\mathbf{D}_{\text{3D}}$.

---

## 5. Evaluation Protocol & Generalization Metrics

### Standardized Closed-Loop Rollout Execution
- **Control Frequency**: Closed-loop policy control runs at $20\,\text{Hz}$ interacting with the MuJoCo Franka Panda environment ($500\,\text{Hz}$ physics solver, $dt=0.002\,\text{s}$).
- **Episode Budget**: $N=20$ evaluation episodes per task initialized with unobserved pseudo-random seeds disjoint from training demonstration seeds ($\mathcal{S}_{\text{test}} \cap \mathcal{S}_{\text{train}} = \emptyset$).
- **Maximum Horizons**: $H_{\max}=400$ steps for single-stage manipulation (D1–D4), $H_{\max}=800$ steps for long-horizon sequences (D5).

### Quantitative Generalization Metrics
1. **Task Success Rate (SR)**:
   $$\text{SR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\mathbf{s}_T^{(i)} \in \mathcal{C}_{\text{aff}})$$
2. **Physical Robustness Index (PRI)**:
   Isolates the exact fraction of nominal capability preserved under environmental perturbations:
   $$\text{PRI}(\mathcal{P}) = \frac{\text{SR}(\mathcal{P})}{\text{SR}(\text{Nominal})} \times 100\%$$
3. **Subgoal Completion Ratio (SCR)**:
   For D5 multi-stage sequences, tracks fractional milestone achievement:
   $$\text{SCR} = \frac{1}{N M}\sum_{i=1}^N \sum_{m=1}^M \mathbb{I}(\text{Subgoal } m \text{ reached})$$
4. **Force Violation Rate (FVR)**:
   Percentage of rollouts terminating prematurely due to excessive contact reaction forces ($F_N > 50\,\text{N}$) or kinematic jamming.

---

## 6. SOTA Baseline Benchmark Results

Evaluated over 20 test rollouts per task under **Nominal (In-Distribution)** and **Composite OOD ($\mathcal{P}_{\mathrm{comp}}$)** perturbation conditions:

| Task Domain | Diffusion Policy | | OpenVLA (7B) | | SmolVLA (450M) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Nominal** | **OOD** | **Nominal** | **OOD** | **Nominal** | **OOD** |
| **D1: Industrial Precision** | 86.4% | 14.2% | 81.0% | 11.5% | 74.2% | 8.5% |
| **D2: Culinary Operations** | 91.2% | 23.0% | 88.5% | 18.2% | 82.0% | 14.0% |
| **D3: Domestic Tabletop** | 94.0% | 31.5% | 92.0% | 27.0% | 87.5% | 21.0% |
| **D4: Laboratory Sciences** | 83.5% | 9.8% | 78.4% | 8.0% | 71.0% | 6.2% |
| **D5: Long-Horizon Sequences**| 72.0% | 6.5% | 67.2% | 4.2% | 59.5% | 3.0% |
| **Aggregate Average** | **85.4%** | **17.0%** | **81.4%** | **13.8%** | **74.8%** | **10.5%** |

### Key Failure Diagnostics:
- **Visual Token Compression in Edge VLAs**: While SmolVLA achieves $>50\,\text{Hz}$ edge inference, its visual token reduction (64 tokens) hinders fine-grained visual servoing in sub-centimeter insertions (D1, D4).
- **Contact Slippage vs. Subtask Planning**: In long-horizon tasks (D5), AtomVLA’s predictive world model prevents subtask sequencing errors; however, when surface friction $\mu \le 0.25$, open-loop gripper commands drop objects during lifting phases.
- **Kinematic Jamming**: High-stiffness impedance control causes peg jamming under minor rotational misalignments in industrial assembly.

---

## 7. Quickstart Guide

### Installation
```bash
git clone https://github.com/s-elim/AFFORD_Bench.git
cd AFFORD_Bench
pip install -e .
```

### Synthesize Demonstrations (100% Local, Zero API Cost)
```bash
# Generate 50 expert demonstrations for an industrial task
python scripts/generate_all_150_tasks.py --domain industrial --num-demos 50

# Generate 50 expert demonstrations for tabletop tasks
python scripts/generate_all_150_tasks.py --domain tabletop --num-demos 50
```

### Render Qualitative Visualizations
```bash
python scripts/generate_paper_figures.py
```

### Benchmark Policies
```bash
# Run closed-loop policy evaluation across all 5 perturbation suites
python scripts/evaluate_policy.py --task industrial_gear_mesh --model diffusion_policy --episodes 20
```

---

## 8. Repository Layout

```text
AFFORD_Bench/
├── README.md                           # Master documentation & protocols
├── setup.py                            # Pip installable package (afford_pro)
├── afford_pro/
│   ├── __init__.py                     # API: make(), list_tasks(), get_oracle()
│   ├── registry.py                     # Master 150-task & oracle registry
│   ├── envs/                           # Gymnasium environments
│   │   ├── base_franka_env.py          # Franka Panda MuJoCo base env with 3D metric depth
│   │   ├── industrial/                 # Domain 1: Industrial Precision Assembly (Tasks 001–030)
│   │   ├── kitchen/                    # Domain 2: Culinary Operations (Tasks 031–060)
│   │   ├── tabletop/                   # Domain 3: Domestic Manipulation (Tasks 061–090)
│   │   ├── lab/                        # Domain 4: Laboratory Sciences (Tasks 091–120)
│   │   └── long_horizon/               # Domain 5: Long-Horizon Sequences (Tasks 121–150)
│   ├── oracles/                        # Deterministic Closed-Loop Algorithmic Solvers
│   ├── perturbations/                  # Affordance-Grounded Perturbation Engine (Spatial, Physics, Visual)
│   └── datasets/                       # Universal Robomimic HDF5 dataset writer
├── AFFORD-PRO_RSS'27/
│   └── paper-template-latex/           # RSS 2027 Stage 1 Extended Abstract LaTeX source
│       ├── main.tex                    # Strict 6-page manuscript source
│       ├── references.bib              # Verified bibliography
│       ├── IEEEtran.cls                # Official RSS template class
│       └── figures/                    # PDF and PNG publication figures
└── scripts/
    ├── generate_all_150_tasks.py       # Parallel batch synthesis pipeline
    ├── generate_paper_figures.py       # High-resolution multi-modal visualization generator
    ├── evaluate_policy.py              # Closed-loop evaluation harness
    └── run_benchmark_suite.py          # Benchmark runner & table formatter
```

---

## 9. Citation

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
