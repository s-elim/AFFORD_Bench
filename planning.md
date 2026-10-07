# AFFORD-Bench: Master Strategic Roadmap & Reviewer Action Plan

**Target Conference**: Robotics: Science and Systems (RSS) 2027 (Athens, Greece)  
**Track**: Two-Stage Shepherding Process (Stage 1 Extended Abstract $\rightarrow$ Stage 2 Extension Charter)  
**Paper Title**: *AFFORD-Bench: An Affordance-Grounded Benchmark and Generalization Evaluation for Robot Learning*  
**Paper ID**: `[RSS-2027-0150]` (Anonymized)

---

## 1. Key Deadlines & The Two-Stage Shepherding Process

RSS 2027 replaces the traditional single-deadline model with a **two-stage collaborative development cycle**:

| Milestone | Date (AoE) | Objectives & Deliverables |
| :--- | :--- | :--- |
| **Stage 1 Submission** | **December 4, 2026** | Strict 6-page Extended Abstract (up to 5 content pages + 1 reference page). Fully fleshed out problem statement, hypotheses, benchmark formulation, and preliminary calibration. |
| **Stage 1 Decision** | **February 5, 2027** | Reject or Invitation to Rebuttal. |
| **Rebuttal Deadline** | **February 12, 2027** | 1-week response to Area Chair (AC) and reviewer feedback. |
| **Stage 2 Charter Notification**| **February 26, 2027** | Finalized **Extension Charter** issued by AC outlining mandatory experiments and analyses. |
| **Final Paper Submission** | **April 16, 2027** | Strict 8-page full paper (excluding references) reporting all charter executions. |
| **Final Acceptance** | **April 30, 2027** | Formal notification of acceptance. |
| **Camera-Ready Copy** | **May 14, 2027** | De-anonymized final PDF with public reviewer/chair signatures. |
| **Conference Presentation**| **July 6–11, 2027** | In-person presentation in Athens, Greece. |

---

## 2. Reviewer Perspective: Honest Pre-Submission Assessment

### Strengths That Boost the Score
1. **Unrivaled Timeliness**: Directly addresses the evaluation crisis exposed by LIBERO-PRO and LIBERO-X (overfitting to visual landmarks, lack of physical grounding).
2. **Benchmark Scale**: 210 physical tasks across 7 specialized domains $\times$ 35 evaluation suites (10,500 demonstrations, $>2.2\times 10^6$ timesteps).
3. **Policy-Agnostic Design**: Built for *any* policy paradigm—not just Diffusion Policy (DP) or Vision-Language-Action (VLA) models:
   - Classical Imitation Learning & Behavior Cloning (MLP BC, BeT, IBC)
   - Action-Chunking Visuomotor Policies (Diffusion Policy, ACT)
   - 3D Spatial & Point-Cloud Policies (DP3 via aligned 3D metric depth)
   - Vision-Language-Action Models (OpenVLA, SmolVLA, AtomVLA, Octo)
   - Offline Reinforcement Learning (CQL, IQL, TD3+BC)
4. **Hierarchical 5-Level Perturbation Framework (L1–L5)**: Progressive difficulty ladder isolating spatial, physical, visual, and compound chaining breakdown frontiers.
5. **Zero External API Cost**: Local C++ MuJoCo physical engine running at 40–50 FPS on standard workstations.

---

### The 3 Critical Reviewer Vulnerabilities (And How We Overcome Them)

| # | Vulnerability Identified | Reviewer Critique | Preemptive Defense & Mitigation Strategy |
| :-: | :--- | :--- | :--- |
| **1** | **Simulation Grasping Mechanics** | *"Gripper closure welds the object to the TCP. If objects are kinematically attached, Coulomb friction $\mu$ cannot cause physical grasp slippage during transport."* | **Transparent Physics Boundary**: State explicitly that kinematic attachment ensures deterministic demonstration generation across all 210 tasks, whereas **docking, peg insertion, gear meshing, sliding, drawer articulation, and collision constraints are fully governed by MuJoCo contact physics**. In the Stage 2 Charter, commit to compliance wrench thresholds and dynamic slip evaluations. |
| **2** | **Stage 1 Baseline Data** | *"Table II has 100% Oracle and 0% Random, but lists 'Calibrating' for BC and Diffusion Policy. Where is empirical proof that policies learn?"* | **Preliminary Baseline Training**: Train a lightweight State BC and State Diffusion Policy on **Domain 1 (Industrial)** and **Domain 3 (Tabletop)** for 20 epochs. Report real numbers (e.g., 60–80% nominal, dropping to 10–20% OOD) in Table II. |
| **3** | **Sim-to-Real Hardware Transfer** | *"Is there real-world robot hardware transfer, or is this purely synthetic?"* | **Dedicated Extension Charter Work-Package**: Frame the physical Franka Emika Panda workstation pilot (gear meshing & pipette docking) as the centerpiece of the Stage 2 Extension Charter. |

---

## 3. Step-by-Step Action Roadmap to Publication

```
                                  MASTER EXECUTION ROADMAP
                                  
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ PHASE 1: STAGE 1 SUBMISSION READINESS (Current Priority - Dec 4, 2026)                                 │
  ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ [x] 1.1 Complete 210-task dataset synthesis (10,500 demonstrations, Robomimic HDF5 format)             │
  │ [x] 1.2 Verify dataset integrity (shapes, dual RGB, 3D metric depth, proprioception)                   │
  │ [x] 1.3 Implement 5-level hierarchical perturbation engine (L1-L5) & 35 evaluation suites              │
  │ [x] 1.4 Redesign main.tex to strict 6-page RSS format (5 pages content + 1 page references)            │
  │ [x] 1.5 Preserve strict anonymity (Author names omitted, Paper ID [RSS-2027-0150], zero external links)│
  │ [x] 1.6 Launch interactive project page (docs/index.html) with 7-domain video demonstrations           │
  │ [x] 1.6b Deliver minimalist, high-impact benchmark comparison figure (LIBERO-PRO vs LIBERO-X vs Ours) │
  │ [ ] 1.7 Train preliminary State-BC & State-Diffusion baselines on Domain 1 (Industrial) & 3 (Tabletop) │
  │ [ ] 1.8 Run evaluate_id.py & evaluate_ood.py to populate preliminary empirical numbers in Table II     │
  │ [ ] 1.9 Recompile final publication-grade main.pdf and freeze Stage 1 submission package               │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ PHASE 2: REBUTTAL & CHARTER SHAPING (Feb 5 – Feb 26, 2027)                                             │
  ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ [ ] 2.1 Analyze Stage 1 Metareview and reviewer comments                                               │
  │ [ ] 2.2 Submit targeted rebuttal addressing contact dynamics, asset variety, and protocol clarity      │
  │ [ ] 2.3 Receive and agree upon finalized Area Chair Extension Charter                                   │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ PHASE 3: STAGE 2 EXTENSION CHARTER EXECUTION (Feb 26 – April 16, 2027)                                 │
  ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ [ ] 3.1 Full Empirical Scaling: Evaluate DP, ACT, OpenVLA, SmolVLA, and DP3 across 35 benchmark suites │
  │ [ ] 3.2 Disentangled Ablations: Plot individual degradation frontiers for spatial, physics, and visual │
  │ [ ] 3.3 Sim-to-Real Pilot: Deploy policies onto physical Franka Emika Panda workstation                 │
  │ [ ] 3.4 Expand manuscript to final 8-page RSS format with comprehensive findings & qualitative failure │
  │ [ ] 3.5 Submit Final Paper by April 16, 2027                                                           │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Immediate Concrete Next Steps (Phase 1.7 – 1.9)

### Step 1: Train Quick State Baseline Policies
Run fast, lightweight state-based training on CPU/GPU for Domain 1 (Industrial) and Domain 3 (Tabletop):
```bash
# 1. Train Diffusion Policy on Industrial domain (10-20 epochs)
python scripts/train_policy.py --policy diffusion --modality state --domain industrial --epochs 20 --batch-size 64

# 2. Train Behavior Cloning (BC) on Industrial domain
python scripts/train_policy.py --policy bc --modality state --domain industrial --epochs 25 --batch-size 64

# 3. Train Diffusion Policy on Domestic Tabletop domain
python scripts/train_policy.py --policy diffusion --modality state --domain tabletop --epochs 20 --batch-size 64

# 4. Train Behavior Cloning (BC) on Domestic Tabletop domain
python scripts/train_policy.py --policy bc --modality state --domain tabletop --epochs 25 --batch-size 64
```

### Step 2: Evaluate In-Distribution & Out-of-Distribution Baselines
```bash
# Evaluate Industrial domain checkpoints under Nominal (ID)
python scripts/evaluate_id.py --checkpoint checkpoints/diffusion_state_industrial_best.pt --domain industrial --num-episodes 10
python scripts/evaluate_id.py --checkpoint checkpoints/bc_state_industrial_best.pt --domain industrial --num-episodes 10

# Evaluate Industrial domain checkpoints under Spatial OOD
python scripts/evaluate_ood.py --checkpoint checkpoints/diffusion_state_industrial_best.pt --domain industrial --perturbation p2_spatial --num-episodes 10
python scripts/evaluate_ood.py --checkpoint checkpoints/bc_state_industrial_best.pt --domain industrial --perturbation p2_spatial --num-episodes 10
```

### Step 3: Update Table II in `main.tex` & Recompile
- Replace `"Calibrating"` entries for D1 and D3 in Table II with the empirical numbers measured above.
- Leave D2, D4, D5, D6, D7 designated as *"Stage 2 Charter Execution"*.
- Compile `main.tex` with `pdflatex` to produce the final, unassailable 6-page submission PDF.

---

## 5. Summary of Benchmark Assets & Verified Files

- **Core Package**: [`afford_pro/`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/afford_pro) (Gymnasium envs, MuJoCo C++, 210 task definitions, perturbation engine).
- **HDF5 Datasets**: [`datasets/*.hdf5`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/datasets) (210 tasks $\times$ 50 demos = 10,500 demos, 2,296,400 steps, ~387 GB).
- **Paper LaTeX Source**: [`AFFORD-Bench_RSS'27/paper-template-latex/main.tex`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/AFFORD-Bench_RSS'27/paper-template-latex/main.tex).
- **Compiled Paper PDF**: [`AFFORD-Bench_RSS'27/paper-template-latex/main.pdf`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/AFFORD-Bench_RSS'27/paper-template-latex/main.pdf) (Strict 6 pages, 0 overfull boxes).
- **Interactive Project Website**: [`docs/index.html`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/docs/index.html).
- **Showcase Videos**: [`docs/videos/*.mp4`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/docs/videos) (7 domain video demos).
- **Master README**: [`README.md`](file:///D:/Selim%20Sarowar/Cursor_Projects/AFFORD-Bench/README.md).
- **Remote Git Repository**: `https://github.com/s-elim/AFFORD_Bench.git` (`main`).
