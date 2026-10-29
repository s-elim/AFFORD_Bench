# AFFORD-Bench: Physical Contact Robustness in Robot Manipulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/RSS-2027-red.svg)](https://roboticsconference.org/)
[![Affiliation](https://img.shields.io/badge/Affiliation-Yeungnam%20University-003876.svg)](https://www.yu.ac.kr/)
[![Benchmarks](https://img.shields.io/badge/Manifests-Full%20(960)%20%7C%20Lite%20(96)-success.svg)](benchmark/)
[![Safety](https://img.shields.io/badge/Safety%20Bound-150%20N%20%7C%20Actuator%20Torques-orange.svg)](docs/BENCHMARK_SPEC.md)

**Yeungnam University** &middot; **Physical Superintelligence**

**Project Webpage:** [https://physical-agi.github.io/AFFORD_Bench/](https://physical-agi.github.io/AFFORD_Bench/)  
**Code Repository:** [https://github.com/Physical-AGI/AFFORD_Bench](https://github.com/Physical-AGI/AFFORD_Bench)

---

## 📌 Overview

**AFFORD-Bench** is a standardized physical manipulation benchmark designed to measure whether robot policies adapt under mid-trajectory contact disturbances without causing dangerous contact force spikes ($F_{\max} > 150\,\text{N}$) or joint motor torque saturation.

Every **Dynamic** rollout is counterfactually matched to a **Base** nominal control sharing identical initial scene layout, random seeds, and executed action prefix $a_{0:t^*}$.

### 🚀 Key Advantages over Vision-Only Benchmarks (e.g. LIBERO-MAX)

| Feature | AFFORD-Bench (Ours) | LIBERO-MAX | robomimic | LIBERO-PRO |
| :--- | :---: | :---: | :---: | :---: |
| **Tasks** | **240** (D1–D8) | 130 | 7 | 100 |
| **Mid-Trajectory Perturbations** | **✓ Paired Prefix Replay** | ✓ Paired Prefix Replay | ✗ Static only | ✗ Static OOD only |
| **Actuator Motor Torques ($\boldsymbol{\tau}_{\text{act}}$)** | **✓ 7-DoF @ 20 Hz** | ✗ None (pixels only) | ✗ None | ✗ None |
| **Contact Force Wrench ($F_{\text{contact}}$)** | **✓ Real-time 3D** | ✗ None | ✗ None | ✗ None |
| **Physical Force Ceiling ($F_{\max} \le 150\,\text{N}$)** | **✓ Evaluated** | ✗ None | ✗ None | ✗ None |
| **Compliant Dynamic Success ($S_{\text{safe}}$)** | **✓ Enforced** | ✗ None (kinematic only) | ✗ None | ✗ None |
| **Frozen Deterministic Manifests** | **✓ Full + Lite Track** | ✓ Full + Lite Track | ✗ None | ✗ None |

---

## 🏗️ Benchmark Scale

* **240 Manipulation Tasks**:
  * **D1 Industrial Assembly**: 30 tight-tolerance peg, shaft, keyway, and bushing tasks.
  * **D2 Kitchen Manipulation**: 30 multi-object food prep and utensil handling tasks.
  * **D3 Tabletop Sorting**: 30 geometric block, tray, and container rearrangement tasks.
  * **D4 Laboratory Automation**: 30 centrifuge tube, pipette, and decanting tasks.
  * **D5 Long-Horizon Assembly**: 30 multi-stage gear, pulley, and transmission chains.
  * **D6 Precision Chaining**: 30 multi-step cleanroom and electronics socket tasks.
  * **D7 Service Chaining**: 30 long-horizon meal service and condiment clearing tasks.
  * **D8 Frontier Reactive Perturbations**: 30 mid-trajectory physical displacements, bayonet twist locks, and spring resistance mechanisms.
* **Evaluation Manifests**:
  * `benchmark/manifests/afford_bench_full.json`: **960 paired cases** (4 seeds per task).
  * `benchmark/lite/afford_bench_lite.json`: **96 paired cases** (3 balanced tasks per domain D1–D8).

---

## ⚡ Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Physical-AGI/AFFORD_Bench.git
cd AFFORD_Bench

# 2. Install dependencies in virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# 3. Validate benchmark schemas & manifests
python scripts/validate_manifests.py

# 4. Evaluate paired dynamic cases (Lite Track)
python scripts/evaluate_paired_dynamic.py \
    --manifest benchmark/lite/afford_bench_lite.json \
    --out results/paired_lite.json
```

---

## 📊 Core Metric: Compliant Dynamic Success ($S_{\text{safe}}$)

A trial is scored as a compliant success if and only if the policy achieves the goal while remaining within physical torque and force limits:

$$S_{\text{safe}} = S_{\text{dyn}} \land (F_{\max} \le 150\,\text{N}) \land \left(\forall j: |\tau_j| \le \tau_{\max, j}\right)$$

where $\tau_{\max} = [87, 87, 87, 87, 12, 12, 12]\,\text{N}\cdot\text{m}$ for the Franka Emika Panda arm, and $150\,\text{N}$ is the ISO/TS 15066 safety threshold for collaborative robots.

---

## 📜 Citation

If you use AFFORD-Bench in your research, please cite:

```bibtex
@article{sarowar2026affordbench,
  title       = {AFFORD-Bench: A Benchmark for Physical Affordance, Compliance, and Reactive Contact Robustness in Robot Manipulation},
  author      = {Sarowar, Selim and Physical Superintelligence Research Team},
  institution = {Yeungnam University},
  journal     = {Robotics: Science and Systems (RSS)},
  year        = {2027},
  url         = {https://github.com/Physical-AGI/AFFORD_Bench}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
