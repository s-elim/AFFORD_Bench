# AFFORD-Bench Specification: Dynamic Robustness & Force Compliance

**Status**: Released Benchmark Specification, Version 1.0.0.  
This document specifies the evaluation tracks, matched-pair counterfactual protocol, force-torque safety constraints, and statistical reporting standards for **AFFORD-Bench**.

---

## 1. Evaluation Tracks

| Track | Matched Pairs | Total Scored Rollouts | Composition | Primary Use |
| :--- | :---: | :---: | :--- | :--- |
| **Full** | 960 | 1,920 | 240 tasks (D1–D8, 4 pairs/task) | Comprehensive policy benchmarking & archival reporting |
| **Lite** | 96 | 192 | 24 tasks (3 per domain D1–D8, 4 pairs/task) | Rapid iteration, CI validation, hyperparameter sweeps |

- **Frozen Manifests**:
  - Full Track: [`benchmark/manifests/afford_bench_full.json`](../benchmark/manifests/afford_bench_full.json)
  - Lite Track: [`benchmark/lite/afford_bench_lite.json`](../benchmark/lite/afford_bench_lite.json)
  - Validation: verified against draft 2020-12 schemas in [`schemas/manifest.schema.json`](../schemas/manifest.schema.json).
- **Lite Subset Contract**:
  - Deterministically selected with seed `20261029`.
  - Preserves exact domain proportions (3 tasks per domain across all 8 domains).

---

## 2. Matched Base and Dynamic Counterfactual Protocol

Every evaluation case defines two paired arms sharing the same random seed, initial simulator state $\mathbf{s}_0$, language instruction, and executed action prefix $a_{0:t^*}$:

$$\text{Base (Control):} \quad \mathbf{s}_0 \xrightarrow{a_{0:t^*}} \mathbf{s}_{t^*} \xrightarrow{\text{nominal continuation}} \text{Outcome}$$

$$\text{Dynamic (Intervention):} \quad \mathbf{s}_0 \xrightarrow{a_{0:t^*}} \mathbf{s}_{t^*} \xrightarrow{\mathbf{Event} \;\; \Pi_{t^*, \Delta\mathbf{x}}} \mathbf{s}'_{t^*} \xrightarrow{\text{policy adapts}} \text{Outcome}$$

1. **Exact Pre-Event Replay**: The policy executes nominally up to trigger point $t^*$. The recorded action prefix $a_{0:t^*}$ is strictly replayed in the Dynamic arm, ensuring zero stochastic drift before intervention.
2. **Trigger Timing**:
   - `pre_grasp_proximity`: Triggers when the gripper descends within $8\,\text{cm}$ of the manipulated workpiece.
   - `mid_approach_step`: Triggers at step $t^* = T_{\text{approach}} / 2$.
3. **Exogenous Intervention**:
   - Workpiece displacement: $\Delta \mathbf{x} = [\Delta x, \Delta y, 0]$ with $|\Delta \mathbf{x}| = 3.0\,\text{cm}$ along canonical axes (`+x`, `-x`, `+y`, `-y`).
   - Velocities zeroed ($\mathbf{v}_{\text{obj}} \leftarrow \mathbf{0}$) to maintain stability.
   - Domain 8 frontier events also include impulse wrenches and goal fixture shifts.

---

## 3. Physical Force & Actuator Torque Compliance (The Key Advantage)

A critical limitation of kinematic-only benchmarks (such as LIBERO-MAX) is that they evaluate policies purely on visual and geometric completion: if an arm drives an object into a fixture with infinite force or binds an insertion peg with saturating torques, kinematic simulators score it as a "success".

AFFORD-Bench measures **Compliant Dynamic Robustness**:

$$\mathbf{S}_{\text{compliant}} = \mathbf{S}_{\text{dynamic}} \;\land\; (F_{\max} \le 150\,\text{N}) \;\land\; (\forall j, |\tau_j| \le \tau_{\max, j})$$

### Safety Envelope:
1. **Cartesian Contact Force Threshold**:
   - $F_{\max} \le 150.0\,\text{N}$ (ISO/TS 15066 transient/quasi-static contact limit for collaborative robotics).
2. **Franka Emika Panda Actuator Torque Limits**:
   $$\boldsymbol{\tau}_{\max} = [87.0, \; 87.0, \; 87.0, \; 87.0, \; 12.0, \; 12.0, \; 12.0]\,\text{N}\cdot\text{m}$$

Any episode that achieves the geometric goal but breaches $F_{\max} > 150\,\text{N}$ or saturates actuator torques is classified as a **Safety Violation**.

---

## 4. The Eight Evaluation Domains (D1–D8)

| Domain | Name | Focus & Constraints | Clearance |
| :--- | :--- | :--- | :---: |
| **D1** | Industrial Precision | Shaft, gear, and pin insertions into machined jig bores | $\pm 2.0\text{--}2.5\,\text{mm}$ |
| **D2** | Culinary Operations | Patty stacking, cheese topping, and bun crowning without slip | Surface Contact |
| **D3** | Domestic Tabletop | Multi-color block sorting among active workspace clutter | $\pm 8.0\,\text{mm}$ |
| **D4** | Laboratory Sciences | Centrifuge micro-tube docking and pipette aspiration | $\pm 2.0\,\text{mm}$ |
| **D5** | Long Horizon | Multi-stage transmission and burger assembly (up to 6 goals) | Stage-Locked |
| **D6** | Precision Chaining | Shaft $\to$ bearing $\to$ clamp compound alignment | Compound |
| **D7** | Service & Assembly | Food tray clearing, condiment placement, and surface wiping | Multi-Phase |
| **D8** | Frontier Complex | Reactive mid-trajectory perturbations, tool use, and detents | $\pm 3.0\text{--}5.0\,\text{mm}$ |

---

## 5. Statistical Reporting & Metrics

For every evaluated policy, benchmark reports must provide:

1. **Paired Outcome Quadrants**:
   - **Preserved Success $(1, 1)$**: Succeeds in both Base and Dynamic arms.
   - **Event-Associated Regression $(1, 0)$**: Succeeds in Base, but fails after the dynamic disturbance.
   - **Change-Associated Recovery $(0, 1)$**: Fails in Base, but succeeds in Dynamic.
   - **Persistent Failure $(0, 0)$**: Fails in both arms.
2. **Conditional Regression Rate**:
   $$\text{CRR} = \frac{\text{count}(1, 0)}{\text{count}(1, 1) + \text{count}(1, 0)}$$
3. **Paired Difference**:
   $$\Delta \text{SR} = \text{SR}_{\text{dynamic}} - \text{SR}_{\text{base}} \quad (\text{percentage points})$$
4. **95% Bootstrap Confidence Intervals**:
   - Calculated over 2,000 stratified bootstrap resamples of matched pairs.
5. **Force Telemetry**:
   - Mean peak contact force $\bar{F}_{\text{base}}$ vs. $\bar{F}_{\text{dynamic}}$.
   - Force violation rate ($F_{\max} > 150\,\text{N}$).
   - Compliant Dynamic Success Rate.

---

## 6. Comparison with Existing Manipulation Benchmarks

| Dimension | LIBERO / LIBERO-PRO | LIBERO-MAX | Robomimic | AFFORD-Bench |
| :--- | :---: | :---: | :---: | :---: |
| **Task Scope** | 130 static tasks | 8,000 paired shifts | 6 environments | **240 physical tasks (D1–D8)** |
| **Evaluation Mode** | Static ($t=0$) | Dynamic ($t^*$) | Static ($t=0$) | **Both Static (L1–L5) & Dynamic** |
| **Force Sensing** | ❌ None | ❌ None | ❌ None | ✅ **3-D $F_{\text{contact}}$ vectors** |
| **Actuator Torques** | ❌ None | ❌ None | ❌ None | ✅ **7-D $\boldsymbol{\tau}_{\text{act}}$ channels** |
| **Safety Violations** | ❌ None | ❌ None | ❌ None | ✅ **$F_{\max} > 150\,\text{N}$ & $\boldsymbol{\tau}_{\max}$ checks** |
| **Clearance Tiers** | Coarse ($>15\,\text{mm}$) | Coarse ($>15\,\text{mm}$) | Coarse ($>10\,\text{mm}$) | **Millimeter-level ($\pm 2.0\,\text{mm}$)** |
| **Friction / Slip** | Kinematic drag | Kinematic drag | Soft penalty | **Elliptic friction cones (impratio 10)** |
