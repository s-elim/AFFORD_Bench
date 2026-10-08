# Frontier complex manipulation: specification of Domain 8 (Tasks 211–240)

Status (2026-10-08): **Stage 2 proposal.** No Domain 8 asset, task definition, oracle, or success predicate
exists in the code yet. The only implemented part is the reactive-perturbation probe of Section 5
(`scripts/test_reactive_perturbation.py`), which runs on existing Stage 1 tasks. This document turns
`planning.md` §6 into specifications that can be implemented and checked. It also states which parts require
capabilities the Stage 1 simulator does not have.

## 0. Constraints inherited from the Stage 1 stack

These limits decide which proposals can be built as written. A capability marked **new** below is a
prerequisite, not a detail.

| Quantity | Stage 1 value | Consequence for Domain 8 |
| --- | --- | --- |
| Action space | TCP position increment (cap 2.5 cm per 50 ms step), hand yaw (0.05 rad/step), gripper opening; hand points down, roll and pitch are not executed | Tilted tool strokes (spatula rake, edge grasps) need a fixed rake built into the tool or **new** roll/pitch control |
| Workspace clamp | x ∈ [0.20, 0.80] m, y ∈ [−0.50, 0.50] m, z ∈ [0, 0.70] m | "Beyond reach" means beyond the clamp (x > 0.80 m), not beyond the Panda's kinematic limit |
| Gripper | Parallel jaw, 0–40 mm per finger, about 35 N per finger, slew-limited to 0.05 m/s per finger | Thin flat bodies (< 6 mm) cannot be pinched from above; tools need a handle 10–60 mm wide |
| Wrist yaw range | Joint 7 limits keep commanded hand yaw within about ±2.0 rad in the oracle | Turns beyond about 180° need a release, counter-rotation, and regrasp |
| Sensing | RGB (workcell, wrist), optional metric depth, joint and TCP proprioception; no force-torque or tactile channel | Force-conditioned success (for example F_z ≥ 20 N) must be evaluated from simulator state, and a policy can only infer force from vision and joint positions |
| Contact model | Rigid bodies, MuJoCo soft constraints (pads: 4 ms time constant), native CCD with multi-point contact, pyramidal friction (condim 3, or 4 with torsional friction) | Thin-blade wedging and detents rely on solver penetration margins of about 0.1–1 mm; parameters must be validated per task |

## 1. Common notation

- Frames: world W, robot hand H, tool τ, tool tip (action point) A, manipulated body o, fixture F.
  T_XY ∈ SE(3) is the pose of Y in X. The tool's in-hand pose T_Hτ is **measured** after the grasp
  (T_Hτ = T_WH⁻¹ T_Wτ), not assumed, because frictional grasps slip.
- Contact set at state s: C(s) = {(p_i, n_i, f_i)} with contact point p_i, unit normal n_i (from the first to the
  second body), and force f_i = f_{n,i} n_i + f_{t,i}.
- Coulomb friction cone with coefficient μ: K_μ(n) = {f : ‖f − (f·n)n‖ ≤ μ f·n, f·n ≥ 0}. MuJoCo combines a
  contact pair by the maximum of the two friction coefficients and linearizes K_μ as a pyramid for condim 3. A
  contact *sticks* when f_i lies strictly inside K_μ, and *slides* on its boundary.
- Physical success, as in Stage 1 (Sec. III): the body is released, supported, at rest
  (‖v‖ < 0.02 m/s, ‖ω‖ < 0.2 rad/s), within (ε_xy, ε_z, ε_θ) of its goal. In addition there is a family-specific
  predicate Φ_k(s) below, and a lift requirement h_min wherever prehension is part of the task.
- Tolerances: Domain 8 uses the precision tier ε_xy = ε_z = 5 mm and ε_θ = 10° unless a task states otherwise.
  The tier is looser than D1/D4 because each task chains a non-prehensile or tool stage before the final
  placement. Joint-space predicates (angles of articulated parts) use ε_q = 3°.

## 2. Family A: tool-mediated manipulation (Tasks 211–218)

The tool turns a robot action into an effect at the tool tip A. The reference frame for the task is
T_WA = T_WH T_Hτ T_τA, and a tool stage succeeds only if the **tool-object** contact, not the hand, produces the
object motion. Formally, during the tool stage, contacts between the hand or fingers and o must be absent:
C_{hand,o}(s_t) = ∅.

**A1 L-hook distal retrieval.** The object rests at x ∈ [0.83, 0.90] m, outside the workspace clamp, so it
cannot be grasped directly. The hook (handle 20 mm, shaft 0.25 m, hook 60 mm at 90°) lies within reach.
- *Contact manifold*: the inner face of the hook against the far side of o, a line contact of length equal to
  the overlap height.
- *Pulling condition* (quasi-static sliding of o on the table, coefficient μ_s): the hook force f_h must
  exceed the sliding resistance, ‖f_h,xy‖ > μ_s m g. It must also keep o from rotating out of the hook: the
  line of action of f_h passes within the hook's inner width of the object's center of friction.
- *Success*: Φ_A1 = (o inside the reachable region x ≤ 0.75 m) ∧ (o later grasped and placed at the goal).
- *Metrics*: retrieval distance and the number of hook re-engagements.

**A2 Spatula wedging.** A flat body (patty 68 × 18 mm, or a 4 mm wafer) lies flush on the table. The spatula has
a 1.5 mm blade with a fixed 15° rake built into the handle, because hand pitch is not commanded.
- *Contact manifolds*: blade edge against the table (line), then blade top face against the body's underside
  (face).
- *Wedging condition*: the blade must stay on the table, with normal force f_{n,table} ≥ f_min (2 N), while it
  advances. The body slips onto the blade instead of being pushed when the blade-body friction is lower than the
  body-table friction, μ_bo < μ_ot. The edge also needs the vertical wedge force f_n,bo cos α to lift it. Stage 2
  assets set μ_ot = 1.0 and μ_bo = 0.3.
- *Compliance*: Stage 1 has no impedance mode. The policy commands Δz = −2 mm below the table contact, and the
  servo's finite stiffness produces the pressing force. It must be verified to stay below the force-violation
  threshold F_max = 150 N.
- *Success*: Φ_A2 = (body supported only by the blade for ≥ 0.5 s) ∧ (placed at the goal with tilt ≤ ε_θ).

**A3 Hex-key socketing.** A hex key or socket engages a recessed hex bolt with a vertical axis, so the turn is
a hand yaw. The bolt is a hinge joint with friction loss τ_f = 0.3 N·m and a torque-limited detent at every 60°.
- *Contact manifold*: six line contacts between the key's hex prism and the socket, which exist only if the
  angular alignment satisfies |θ_key − θ_bolt| mod 60° ≤ δ, where δ = arcsin(c / r_hex) for clearance c = 0.2 mm.
- *Torque transfer*: a bolt turn of Δθ needs the in-hand key not to slip, i.e. the grasp's torsional friction
  must satisfy τ_f < μ_t f_grip r_contact. If it does not, the key rotates in the fingers, which the measured
  T_Hτ exposes.
- *Success*: Φ_A3 = (θ_bolt − θ_0 ≥ 90° − ε_q).

| ID | Task | Tool | Needs new capability |
| --- | --- | --- | --- |
| 211 | Hook the spur gear in from x = 0.86 m and place it on the pallet | L-hook | asset |
| 212 | Hook the conical tube rack in from x = 0.84 m | L-hook | asset |
| 213 | Hook a mug by its handle from behind the coffee machine | L-hook | asset |
| 214 | Lift a flush patty with the spatula and place it on the bun | spatula (15° rake) | asset |
| 215 | Lift a 4 mm wafer from the tray with the spatula | spatula | asset |
| 216 | Turn a hex bolt 90° with an Allen key | hex key | asset, hinge joint |
| 217 | Turn a recessed socket bolt 180° (one regrasp) | socket wrench | asset, hinge joint |
| 218 | Sweep three pins into a recessed tray with a pusher bar | bar | asset |

## 3. Family B: non-prehensile manipulation (Tasks 219–226)

**Pushing model.** Under quasi-static planar pushing, the body's motion follows the limit surface of its
support pressure distribution. A point push sticks when the pusher force lies inside the *motion cone*, the image
of the pusher friction cone K_{μ_p} under the limit-surface map. A push outside the motion cone slides along the
body. Stage 2 assets fix μ_p (pusher-body) and μ_s (body-support) per task, so the motion cone, and hence the
set of stable pushes, is known when the oracle is designed.

**B1 Table-edge singulation.** A thin plate (e.g. 80 × 60 × 4 mm) cannot be pinched from above.
- *Push target*: overhang fraction ρ = A_over/A ∈ [0.3, 0.45]. For a uniform plate this is stable, since the
  center of mass stays over the table while ρ < 0.5. Success needs ρ ≥ 0.3 without the plate falling.
- *Grasp*: grasping the exposed lip needs a pinch across the plate's thickness, with a horizontal hand
  approach. That requires **new** wrist roll/pitch control. Without it, Stage 2 can evaluate the push stage alone
  (Φ_B1 = ρ ∈ [0.3, 0.45] ∧ plate at rest).

**B2 Fixture-assisted pivoting.** A cylinder lying on its side (pipette tube, bolt) is pushed against a vertical
fixture of height h_F and pivots about its edge contact.
- *Pivot condition*: the pushing contact sits above the fixture contact. The body tips upright when the
  pusher's moment about the fixture edge exceeds the gravity moment m g r_⊥. Meanwhile the fixture contact must
  not slide (f inside K_{μ_F}) and the table contact must slide (f on the boundary of K_{μ_t}). Assets use
  μ_F = 1.0 and μ_t = 0.3.
- *Success*: Φ_B2 = tilt from upright ≤ ε_θ, then a grasp-and-place as in Stage 1.

**B3 Corral sweeping.** N = 3–5 pins scattered within 15 cm are swept into a recessed tray with a pusher bar.
- *Success*: Φ_B3 = (at least k of N pins inside the tray footprint and at rest). Report the sweep efficiency
  k/N and the number of strokes.

| ID | Task | Needs new capability |
| --- | --- | --- |
| 219 | Push a thin plate to 30–45% overhang (push stage only) | asset |
| 220 | Overhang and grasp a thin plate | asset, wrist roll/pitch |
| 221 | Pivot a lying pipette tube upright against the rack | asset |
| 222 | Pivot a lying bolt upright against the jig and insert it | asset |
| 223 | Push a mug out from under the brew head, then grasp it | none (existing scene) |
| 224 | Slide a cheese slice off the board edge and grasp it | wrist roll/pitch |
| 225 | Sweep 3 pins into the pin feeder | asset (pusher bar) |
| 226 | Sweep 5 coffee pods into the caddy | asset (pusher bar) |

## 4. Family C: multi-axis twist-lock and detent assembly (Tasks 227–233)

All mechanisms keep their axis vertical, so the turn is a hand yaw within the Stage 1 action space. Each
mechanism is an articulated body with a slide joint z (along the axis) and a hinge joint θ.

**Configuration-space manifold.** The admissible set Q ⊂ (z, θ) encodes the mechanism:
- *Bayonet (227–229)*: Q = {θ = 0, z ∈ [z_lock, 0]} ∪ {z = z_lock, θ ∈ [0, 45°]} ∪
  {θ = 45°, z ∈ [z_lock, z_seat]}. This is an L-shaped slot followed by the seat.
  - Geometry: compression Δz = z_lock = −12 mm against a spring k_z = 1.5 N/mm (18 N at full compression), a
    45° turn, then release into the seat.
  - Realization: pins in a slot built from box geoms. The spring is a slide-joint stiffness.
- *Key and deadbolt (230–231)*: the key enters a vertical keyway (blade 2.0 mm, slot 2.0 ± 1.2 mm) to depth
  d ≥ 18 mm. Only then can θ reach 90°, which retracts a latch body on a second slide joint coupled by contact.
  The keyway faces up so that insertion is vertical and the turn is a yaw; a horizontal door lock would need
  **new** roll control.
- *Push-and-turn cap (232–233)*: the cap turns the bottle only if the axial compression satisfies
  F_z = k_c Δz ≥ 20 N. With k_c = 2 N/mm this is Δz ≥ 10 mm, and it engages ratchet teeth (contact geometry).
  Below the threshold the teeth slip.
  - Evaluation: F_z is read from the slide-joint spring force in simulator state. Policies observe only RGB and
    proprioception.
  - A 180° opening turn requires one regrasp within the wrist-yaw range.

**Detents.** MuJoCo has no periodic joint potential, so detents are spring-loaded ball geoms against notched
rings (contact-based). Holding torque is set by the spring preload, and Stage 2 validates it per asset.

**Success.** Φ_C = (the mechanism reaches its terminal configuration within ε_q and ε_z, and stays there for
1 s after release). For 232–233, the predicate also requires that F_z ≥ 20 N held during the turn, read from
the spring force.

| ID | Task |
| --- | --- |
| 227 | Bayonet: push, twist 45°, and seat a filter cartridge |
| 228 | Bayonet: remove the cartridge (reverse path) |
| 229 | Bayonet with a 2-step detent (22.5° then 45°) |
| 230 | Insert the key and turn it 90° to retract the latch |
| 231 | Retract the latch, then slide the bolt open (two bodies) |
| 232 | Push-and-turn cap: open 180° (one regrasp) |
| 233 | Push-and-turn cap: close until the detent clicks |

## 5. Family D: mid-trajectory reactive perturbations (Tasks 234–240)

**Operator.** A perturbation Π_{t_p, Δ} acts at step t_p on the state:
- *displacement*: p_o ← p_o + Δp, v_o ← 0;
- *impulse*: v_o ← v_o + J/m.

The displacement is checked to leave o resting flat without interpenetration; otherwise it is re-drawn.
Domain 8 uses |Δp| = 4 cm along ±x or ±y, and J such that |Δv| = 0.3 m/s.

**Policy classes.** A policy that replans every H steps executes a_{t..t+H−1} = π(s_t).
- Closed-loop control is H = 1.
- Action chunking is H > 1 (ACT and Diffusion Policy commonly execute 8–16 of the predicted steps).
- Open-loop trajectory replay is H = T.

After the perturbation, a chunked policy keeps executing stale actions for (H − t_p mod H) steps, so the
recovery time is bounded below by that number.

**Metrics.**
- Success under perturbation SR_Π, and the **Reactive Recovery Ratio** RRR = SR_Π / SR_nominal.
- Recovery time t_rec: the steps after t_p until the TCP is back within 1 cm (xy) of the displaced object.
- Final xy error at the grasp.
- The **chunk-phase curve** SR_Π(t_p mod H): sweeping t_p across one chunk separates chunking latency from
  visual tracking errors.
- For world models: the prediction error of the post-perturbation state, and whether replanning with the model
  recovers.

**Measured probe (Stage 1 tasks, `scripts/test_reactive_perturbation.py`).** The probe runs the same
privileged approach law at H = 1, 16, and T = 60, with the displacement at t = 30. Results are in
`docs/reviews/reactive_perturbation_2026-10-08.csv`; see the summary below.

Three Stage 1 tasks were probed (`tabletop_sort_red`, `tabletop_sort_blue`, `industrial_pin_to_tray`).
Each run used 3 seeds and ±4 cm along x and y. Displacements that would have left the body unsupported or
interpenetrating were skipped, which left 19 valid displaced episodes per policy:

| Policy | Success, displaced | Success, no displacement (control) | Final xy error, median / max | Recovery steps, median |
| --- | --- | --- | --- | --- |
| Closed loop (H = 1) | 18/19 | 9/9 | 0.0 / 13.4 mm | 13 |
| Chunked (H = 16) | 18/19 | 9/9 | 0.0 / 13.9 mm | 16 |
| Open-loop replay (H = T = 60) | 0/19 | 9/9 | 40.0 / 72.7 mm | never |

Without the displacement all three policies succeed, so the open-loop failures are caused by the displacement
alone. The chunked policy recovers 3 steps later than closed-loop control, within its bound of at most
H = 16 stale steps. That is short enough to succeed in this 60-step approach, so a single mid-chunk displacement
does not separate chunking from closed-loop control here. Tasks 239–240 (chunk-phase sweep, double
displacement) and shorter post-perturbation budgets are the Domain 8 designs meant to separate them. The probe
uses a privileged state-feedback law, not a learned policy, so it validates the test, not any model.

| ID | Task (perturbation) |
| --- | --- |
| 234 | Displace the red cube 4 cm at mid-approach, then sort it |
| 235 | Displace the shaft 4 cm at mid-approach, then insert it |
| 236 | Displace the mug 4 cm, then place it on the serving tray |
| 237 | Impulse on the conical tube during descent, then rack it |
| 238 | Displace the goal fixture (pallet) 4 cm during transport |
| 239 | Displace at the chunk boundary vs mid-chunk (paired, H-sweep) |
| 240 | Displace twice (t = T/3 and 2T/3), then place |

## 6. Inclusion gate and evaluation

A Domain 8 task enters the benchmark only if it passes all of the following:
1. A privileged oracle solves at least 95% of 50 nominal episodes, with the success predicate evaluated on
   simulator state.
2. Null and uniform-random policies solve none of them, and no task is solved at reset.
3. Its contact parameters (friction pairs, spring constants, detent preloads) are validated by a probe like
   `scripts/l4_slip_probe.py`, so that success depends on the stated mechanics and not on solver artifacts such
   as penetration or tunneling.

Domain 8 inherits the L1–L5 suites. L4 is especially informative for Families A–C, because it changes the
friction cones on which the tool, push, and twist conditions above depend.

## 7. Implementation risks (ranked)

1. **Wrist roll and pitch** (Tasks 220 and 224, and any horizontal approach). This changes the action space, so
   Stage 1 learned policies would not transfer.
2. **Articulated mechanisms** (Family C). Detents and ratchets built from contact geometry are sensitive to
   solver margins, so each asset needs a parameter sweep.
3. **Thin-blade wedging** (A2). This needs solver penetration well below the 1.5 mm blade thickness, and may
   need a smaller timestep for the blade contact.
4. **Oracle complexity.** Tool use and pushing need planners with explicit tool frames and motion cones, not
   the Stage 1 pick-and-place oracle.
