# AFFORD-PRO-150: Benchmark, Dataset Generator & Generalization Evaluation Suite for Robot Foundation Models

---

## 1. Top-Venue Publication Strategy

### Target Venues
- **CoRL 2026** (Conference on Robot Learning)
- **RSS 2026** (Robotics: Science and Systems)
- **NeurIPS 2026** (Datasets & Benchmarks Track)
- **ICRA 2027** (IEEE International Conference on Robotics and Automation)

### Proposed Paper Titles
1. **Primary Candidate**:  
   > **"AFFORD-PRO: Benchmarking Generalization in Vision-Language-Action Models Across 150 Diverse Manipulation Tasks Beyond Memorization"**
2. **Alternative A (Affordance Understanding Focus)**:  
   > **"Beyond Memorization: AFFORD-PRO-150 for Stress-Testing Generalization and Affordance Robustness in Robot Foundation Models"**
3. **Alternative B (Fundamental Question)**:  
   > **"Do VLAs Truly Understand Affordances? AFFORD-PRO-150: A Systematic Out-of-Distribution Manipulation Benchmark"**

---

## 2. Why AFFORD-PRO-150 is Significantly Better than LIBERO-PRO

| Capability | LIBERO-PRO (Existing Baseline) | **AFFORD-PRO-150 (Our Benchmark)** | Why It Matters for Top Venues |
| :--- | :--- | :--- | :--- |
| **Task Count** | ~10–40 tasks (mostly small table toys) | **150 Unique Tasks** across 5 distinct domains | 4$\times$ larger coverage; proves generalist capabilities rather than narrow overfit. |
| **Domain Diversity** | Single domestic/kitchen tabletop | **5 Real-World Industrial & Scientific Domains** (Industrial Mfg, Commercial Kitchen, Domestic Tabletop, Laboratory Life-Sciences, Long-Horizon Multi-Stage) | Evaluates transferable cross-domain physical affordances beyond toy kitchen objects. |
| **Sensor Suite** | Single third-person view | **Synchronized Dual-Camera Suite** (Agentic Workcell Overview + Eye-in-Hand Wrist Cam + Metric Depth) | Standardizes realistic multi-view perception used in real Franka Panda / mobile VLA systems. |
| **Perturbation Axes** | 4 dimensions (Object, Position, Semantic, Environment) | **5-Axis Systematic OOD Suite** (+ Physical Dynamics, Adversarial Clutter, Continuous Viewpoint Jitter, Photometric Lighting) | Isolates precisely *where* and *why* VLA policies fail (Visual vs Spatial vs Dynamic). |
| **Kinematics & Speeds** | Abstract velocity steps | **Realistic Franka Panda Dynamics (60 FPS, S-Curve Interpolation)** | Preserves physical realism for sim-to-real transfer. |
| **Data Generation** | Manual human teleoperation or rigid replay | **100% Local Simulation Algorithmic Oracles (500+ FPS)** | Zero external API calls, $0.00 cost, instant batch demo generation on standard hardware. |
| **VLA Compatibility** | Robomimic HDF5 (LIBERO specific) | **Universal Robomimic / LIBERO / OpenVLA Standard HDF5** | Plug-and-play with OpenVLA, Octo, $\pi_0$, Diffusion Policy, and ACT. |

---

## 3. The 150-Task Taxonomy Matrix (5 Domains $\times$ 30 Tasks)

```
                                  AFFORD-PRO-150 TASK TAXONOMY
                                  
  ┌───────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
  │ 1. Industrial & Mfg   │ 2. Culinary & Kitchen │ 3. Tabletop & Domestic│ 4. Lab & Life-Science │ 5. Long-Horizon Tasks │
  │   (Tasks 001 - 030)   │   (Tasks 031 - 060)   │   (Tasks 061 - 090)   │   (Tasks 091 - 120)   │   (Tasks 121 - 150)   │
  ├───────────────────────┼───────────────────────┼───────────────────────┼───────────────────────┼───────────────────────┤
  │ 001. Shaft into bore  │ 031. Patty onto bun   │ 061. Sort red block   │ 091. Pick pipette     │ 121. Coffee brew end  │
  │ 002. Gear spline mesh │ 032. Cheese on patty  │ 062. Sort blue block  │ 092. Transfer testtube│ 122. Multi-layer burger│
  │ 003. Dowel pin press  │ 033. Tomato slice add │ 063. Sort green block │ 093. Tube to rack slot│ 123. 8-stage assembly │
  │ 004. Bearing insertion│ 034. Lettuce top crown│ 064. Open upper drawer│ 094. Uncap tube vial  │ 124. Multi-bin kitting│
  │ 005. Bolt pick & place│ 035. Bun lid placement│ 065. Open lower drawer│ 095. Place centrifuge │ 125. Package boxing   │
  │ 006. Nut threading    │ 036. Mug to espresso  │ 066. Push drawer shut │ 096. Latch centrifuge │ 126. Table clear-off  │
  │ 007. PCB chip insert  │ 037. Press brew button│ 067. Pick ceramic mug │ 097. Pipette to well  │ 127. Table wipe & stow│
  │ 008. Fuse socket seat │ 038. Remove brewed mug│ 068. Mug onto tray    │ 098. Eject pipette tip│ 128. Multi-tool swap  │
  │ 009. Cable harness plug│ 039. Decanter pouring│ 069. Pick flat plate  │ 099. Transfer petri   │ 129. Breakfast prep   │
  │ 010. Relay switch plug│ 040. Water pitcher    │ 070. Stack bowl       │ 100. Open incubator   │ 130. Industrial inspect│
  │ ... (up to 030)       │ ... (up to 060)       │ ... (up to 090)       │ ... (up to 120)       │ ... (up to 150)       │
  └───────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
```

---

## 4. LIBERO-PRO Style 5-Axis Systematic Perturbation Protocol

Every task has one In-Distribution split and five Out-of-Distribution evaluation splits:

```
                            5 SYSTEMATIC GENERALIZATION DIMENSIONS
                                              │
    ┌─────────────────┬───────────────────────┼───────────────────────┬─────────────────┐
    ▼                 ▼                       ▼                       ▼                 ▼
[Object (P1)]   [Spatial (P2)]          [Visual (P3)]           [Clutter (P4)]    [Semantic (P5)]
• Color shifts  • Continuous (x,y)      • Procedural lighting   • 2-4 distractor  • Paraphrasing
• Mesh scaling  • Random yaw (±180°)    • 12 table textures     • False targets   • Multilingual
• Aspect ratio  • Swapped placements    • Specular reflections  • Visual noise    • High/Low level
```

---

## 5. System Architecture (100% Local, Zero Cost)

```
                            AFFORD-PRO-150 ENGINE PIPELINE
                            
    [Task Specification] ──► [Perturbation Engine] ──► [Franka Panda MuJoCo Scene]
                                                               │
                                ┌──────────────────────────────┴──────────────────────────────┐
                                ▼                                                             ▼
                   [Local Algorithmic Oracle]                                    [Local VLA Policy Under Test]
                   • Deterministic Waypoint IK                                   • OpenVLA / Octo / Pi0
                   • Zero API calls ($0.00 cost)                                 • Local PyTorch GPU checkpoint
                   • 500+ FPS headless rollout                                   • env.step(action)
                                │                                                             │
                                ▼                                                             ▼
                   [Robomimic HDF5 Dataset]                                      [Benchmark Metrics (SR, GDI)]
```

---

## 6. Project Directory Layout

```text
AFFORD-PRO-150/
├── planning.md                         # Master research plan & publication blueprint
├── README.md                           # Documentation & quickstart
├── setup.py                            # Pip installable package (afford_pro)
├── afford_pro/
│   ├── __init__.py
│   ├── envs/                           # Gymnasium environments
│   │   ├── base_franka_env.py          # Unified Franka Panda MuJoCo base env
│   │   ├── industrial/                 # Domain 1: Industrial Precision Assembly (30 tasks)
│   │   ├── kitchen/                    # Domain 2: Culinary Operations (30 tasks)
│   │   ├── tabletop/                   # Domain 3: Domestic Manipulation (30 tasks)
│   │   ├── lab/                        # Domain 4: Laboratory Sciences (30 tasks)
│   │   └── long_horizon/               # Domain 5: Multi-Stage Contact (30 tasks)
│   ├── assets/                         # Self-contained MuJoCo XML components & fixtures
│   ├── oracles/                        # 100% Local Algorithmic Demonstration Solvers
│   │   ├── base_oracle.py
│   │   └── industrial_oracle.py
│   ├── perturbations/                  # LIBERO-PRO style perturbation engine
│   │   └── engine.py                   # P1-P5 OOD randomizer
│   ├── datasets/                       # Dataset writers & utilities
│   │   └── hdf5_writer.py              # Universal Robomimic / LIBERO HDF5 exporter
│   └── evaluation/                     # VLA & Policy Evaluation Harness
├── datasets/                           # Output HDF5 demonstration datasets
└── scripts/
    ├── generate_demos.py               # Batch generate demonstrations (headless, fast)
    └── evaluate_policy.py              # Benchmark VLA / policy checkpoint across OOD splits
```

---

## 7. Next Implementation Milestones

1. **Task Suites Expansion**: Implement remaining Domain 1 tasks (gear mesh, dowel pin insertion) and Domain 2 (burger stacking, coffee brewing).
2. **Batch Demo Generation Engine**: Generate 50 demonstrations per task across all splits.
3. **Automated Benchmark Leaderboard Generator**: Script that runs OpenVLA/Octo/Diffusion Policy checkpoints and renders the paper's comparison table automatically.
