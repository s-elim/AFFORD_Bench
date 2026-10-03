# AFFORD-PRO-150: Benchmark, Dataset Generator & OOD Generalization Suite for Robot Foundation Models

[![Target Venues: CoRL / RSS / NeurIPS 2026](https://img.shields.io/badge/Target%20Venues-CoRL%20%7C%20RSS%20%7C%20NeurIPS%202026-blue)](planning.md)
[![100% Simulation Native](https://img.shields.io/badge/MuJoCo-100%25%20Simulation-brightgreen)](https://mujoco.org)
[![Zero External API Cost](https://img.shields.io/badge/API%20Cost-%240.00%20Zero-success)]()
[![Dataset Standard: Robomimic HDF5](https://img.shields.io/badge/Dataset-Robomimic%20HDF5-orange)](https://robomimic.github.io)

**AFFORD-PRO-150** is a next-generation robotics benchmark and demonstration dataset suite specifically engineered to stress-test the out-of-distribution (OOD) generalization and physical affordance understanding of **Vision-Language-Action (VLA) models** (OpenVLA, Octo, $\pi_0$) and imitation learning policies (Diffusion Policy, ACT).

---

## 1. Why AFFORD-PRO-150 Outperforms LIBERO-PRO

| Capability | LIBERO-PRO (Baseline) | **AFFORD-PRO-150 (Our Benchmark)** | Why It Matters for Top Venues |
| :--- | :--- | :--- | :--- |
| **Task Count** | ~10–40 tasks (narrow tabletop toys) | **150 Unique Tasks** across 5 real-world domains | Proves generalist affordance reasoning rather than single-scene memorization. |
| **Domain Diversity** | Single domestic kitchen tabletop | **5 Real-World Industrial & Scientific Domains** (Industrial Precision Mfg, Commercial Kitchen, Domestic Tabletop, Laboratory Sciences, Long-Horizon Multi-Stage) | Stress-tests cross-domain tactile and spatial affordance transfer. |
| **Perception Suite** | Single third-person camera | **Synchronized Multi-View Suite** (Agentic Workcell Camera + Eye-in-Hand Wrist Cam + Metric Depth) | Matches standard multi-camera perception setups used in real Franka Panda deployments. |
| **Perturbation Axes** | 4 dimensions (simple shifts) | **5-Axis Systematic OOD Protocol** (P1 Object, P2 Spatial, P3 Visual, P4 Clutter, P5 Semantic + Combined OOD) | Quantifies the exact Generalization Degradation Index (GDI) across isolated physical axes. |
| **Kinematic Fidelity** | Abstract velocity jumps | **Realistic Franka Panda Dynamics (60 FPS, S-Curve Interpolation)** | Preserves physical smoothness for sim-to-real transfer. |
| **Data Synthesis** | Manual teleoperation / rigid replay | **100% Local Algorithmic Oracles (500+ FPS)** | **Zero external API calls, zero credit cost ($0.00)**, instant offline demonstration collection. |
| **Model Compatibility**| Custom format | **Standard Robomimic / LIBERO / OpenVLA HDF5** | Direct drop-in training for OpenVLA, Octo, Diffusion Policy, and ACT. |

---

## 2. Benchmark Architecture

```
                             AFFORD-PRO-150 ARCHITECTURE
                             
   [Task Specification] ──► [5-Axis Perturbation Engine] ──► [Franka Panda MuJoCo Scene]
                                                                     │
                                    ┌────────────────────────────────┴────────────────────────────────┐
                                    ▼                                                                 ▼
                       [Local Algorithmic Oracle]                                        [Local VLA Policy Under Test]
                       • Deterministic Waypoint IK                                       • OpenVLA / Octo / Pi0
                       • Zero API calls ($0.00 cost)                                     • Local PyTorch GPU checkpoint
                       • 500+ FPS headless rollout                                       • env.step(action)
                                    │                                                                 │
                                    ▼                                                                 ▼
                       [Robomimic HDF5 Dataset]                                          [Benchmark Metrics (SR, GDI)]
```

---

## 3. The 150-Task Taxonomy Matrix (5 Domains $\times$ 30 Tasks)

1. **Domain 1: Industrial Precision Assembly** (Tasks 001–030)
   - Shaft bore seating, gear spline meshing, dowel pin insertion, bearing press, PCB chip insertion, fuse socket seating, cable harness plug.
2. **Domain 2: Culinary & Commercial Kitchen** (Tasks 031–060)
   - Patty stacking from sizzling grill, cheddar cheese placement, tomato slicing/placement, sesame bun crown, espresso cup placement, brew button actuation.
3. **Domain 3: Tabletop & Domestic Manipulation** (Tasks 061–090)
   - Geometric and color bin sorting (red, green, blue), drawer slide actuation, mug caddy placement, dining plate stacking.
4. **Domain 4: Laboratory & Life Sciences** (Tasks 091–120)
   - Pipette rack retrieval, test-tube rack transfer, centrifuge vial loading, petri dish lid transfer.
5. **Domain 5: Long-Horizon Contact Manipulation** (Tasks 121–150)
   - End-to-end breakfast prep, full multi-component transmission assembly, optical QA inspection scan dwell.

---

## 4. Quickstart

### Installation
```bash
git clone https://github.com/your-org/AFFORD-PRO-150.git
cd AFFORD-PRO-150
pip install -e .
```

### Batch Generate Robomimic Demonstrations (Zero API Cost)
```bash
# Generate 50 expert demonstrations for industrial gear meshing across spatial OOD split
python scripts/generate_demos.py --task industrial_gear_mesh --perturbation p2_spatial --num-demos 50

# Generate 50 demonstrations for culinary burger stacking under nominal conditions
python scripts/generate_demos.py --task kitchen_patty_stack --perturbation nominal --num-demos 50
```

### Benchmark Policy or VLA Model
```bash
# Evaluate a single task across all 5 OOD splits
python scripts/evaluate_policy.py --task kitchen_patty_stack --num-episodes 10

# Run complete benchmark suite and output publication Markdown results table
python scripts/run_benchmark_suite.py --domain all --num-episodes 5
```

---

## 5. Directory Layout

```text
AFFORD-PRO-150/
├── planning.md                         # Master research plan & publication blueprint
├── README.md                           # Documentation & quickstart
├── setup.py                            # Pip installable package (afford_pro)
├── afford_pro/
│   ├── __init__.py                     # High-level API: make(), list_tasks(), get_oracle()
│   ├── registry.py                     # Master task & oracle registry
│   ├── envs/                           # Gymnasium environments
│   │   ├── base_franka_env.py          # Unified Franka Panda MuJoCo base env
│   │   ├── industrial/                 # Domain 1: Industrial Precision Assembly
│   │   ├── kitchen/                    # Domain 2: Culinary Operations
│   │   ├── tabletop/                   # Domain 3: Domestic Manipulation
│   │   ├── lab/                        # Domain 4: Laboratory Sciences
│   │   └── long_horizon/               # Domain 5: Multi-Stage Contact
│   ├── assets/                         # Self-contained MuJoCo XML components & fixtures
│   ├── oracles/                        # 100% Local Algorithmic Demonstration Solvers
│   ├── perturbations/                  # LIBERO-PRO style perturbation engine (P1-P5 OOD)
│   ├── datasets/                       # Universal Robomimic / LIBERO HDF5 exporter
│   └── evaluation/                     # VLA & Policy Evaluation Harness
├── datasets/                           # Output HDF5 demonstration datasets
└── scripts/
    ├── generate_demos.py               # Batch generate demonstrations (headless, fast)
    ├── evaluate_policy.py              # Benchmark single task across OOD splits
    └── run_benchmark_suite.py          # Run full suite & format publication table
```
