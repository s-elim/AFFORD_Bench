# AFFORD-Bench Lite Benchmark Track

**AFFORD-Bench Lite** is a frozen, deterministic subset of AFFORD-Bench designed for rapid policy iteration, hyperparameter tuning, and CI testing.

## Specification

- **Total Matched Pairs**: 96
- **Total Scored Rollouts**: 192 (96 Base Control + 96 Dynamic Intervention)
- **Domain Composition**: 3 representative tasks per domain across all 8 domains (D1 through D8):
  - **D1 Industrial Precision**: `industrial_shaft_assembly`, `industrial_gear_mesh`, `industrial_pin_insertion`
  - **D2 Culinary Operations**: `kitchen_patty_on_bun`, `kitchen_cheese_on_patty`, `kitchen_top_bun_crown`
  - **D3 Domestic Tabletop**: `tabletop_sort_red`, `tabletop_sort_green`, `tabletop_sort_blue`
  - **D4 Laboratory Sciences**: `lab_micro_to_centrifuge`, `lab_conical_to_orange_rack`, `lab_conical_to_blue_rack`
  - **D5 Long Horizon Sequences**: `long_horizon_transmission_assembly`, `long_horizon_classic_burger`, `long_horizon_sort_all_blocks`
  - **D6 Precision Chaining**: `chain_shaft_bearing_clamp`, `chain_bearing_then_gear`, `chain_bolt_and_nut`
  - **D7 Service & Assembly**: `service_cheese_and_crown`, `service_patty_cheese_plate`, `service_coffee_set`
  - **D8 Frontier & Reactive**: `frontier_reactive_sort_red`, `frontier_reactive_shaft_seat`, `frontier_reactive_double_displacement`

## Evaluation Protocol

1. **Exact Pre-Event Replay**: Both Base and Dynamic episodes share identical random seeds, initial scene states, and executed pre-event action prefixes $a_{0:t^*}$.
2. **Force & Torque Verification**:
   - $F_{\max} \le 150\,\text{N}$ (ISO/TS 15066 safety threshold).
   - $\boldsymbol{\tau}_{\text{act}} \le [87, 87, 87, 87, 12, 12, 12]\,\text{N}\cdot\text{m}$ (Panda limits).
3. **Scoring**:
   - Headline metric: **Compliant Dynamic Success Rate** ($S_{\text{safe}} = S_{\text{dyn}} \land \neg \text{safety\_violation}$).

## Running Lite Evaluation

```bash
python scripts/evaluate_paired_dynamic.py --tasks representative --max-cases 96
```
