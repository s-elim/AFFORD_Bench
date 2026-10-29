"""Generate deterministic benchmark manifests for AFFORD-Bench (Full and Lite tracks).

Outputs:
  - benchmark/manifests/afford_bench_full.json
  - benchmark/lite/afford_bench_lite.json
  - benchmark/lite/case_index.csv
  - benchmark/lite/README.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.bench.tasks import TASK_MAP
from afford_pro.task_definitions import ALL_150_TASKS, DOMAIN_8_FRONTIER_TASKS

OUTPUT_DIR = ROOT / "benchmark"
MANIFESTS_DIR = OUTPUT_DIR / "manifests"
LITE_DIR = OUTPUT_DIR / "lite"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
LITE_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_MAPPING = {
    "Industrial & Manufacturing": ("D1", "Industrial Precision & Manufacturing"),
    "Culinary Operations": ("D2", "Culinary Operations & Assembly"),
    "Tabletop & Domestic": ("D3", "Domestic & Tabletop Objects"),
    "Laboratory Sciences": ("D4", "Laboratory Sciences & Micro-Manipulation"),
    "Long Horizon": ("D5", "Long-Horizon Sequential Assembly"),
    "Multi-Stage Precision Chaining": ("D6", "Precision Multi-Body Chaining"),
    "Sequential Service & Assembly": ("D7", "Sequential Food & Tray Service"),
    "Frontier Complex Manipulation": ("D8", "Frontier Reactive & Articulated Manipulation"),
}

DIRECTIONS = {
    "+x": [0.03, 0.0, 0.0],
    "-x": [-0.03, 0.0, 0.0],
    "+y": [0.0, 0.03, 0.0],
    "-y": [0.0, -0.03, 0.0],
}

LITE_TASKS = [
    # D1 Industrial
    "industrial_shaft_assembly", "industrial_gear_mesh", "industrial_pin_insertion",
    # D2 Culinary
    "kitchen_patty_on_bun", "kitchen_cheese_on_patty", "kitchen_top_bun_crown",
    # D3 Domestic
    "tabletop_sort_red", "tabletop_sort_green", "tabletop_sort_blue",
    # D4 Laboratory
    "lab_micro_to_centrifuge", "lab_conical_to_orange_rack", "lab_conical_to_blue_rack",
    # D5 Long Horizon
    "long_horizon_transmission_assembly", "long_horizon_classic_burger", "long_horizon_sort_all_blocks",
    # D6 Precision Chaining
    "chain_shaft_bearing_clamp", "chain_bearing_then_gear", "chain_bolt_and_nut",
    # D7 Service Chaining
    "service_cheese_and_crown", "service_patty_cheese_plate", "service_coffee_set",
    # D8 Frontier Complex
    "frontier_reactive_sort_red", "frontier_reactive_shaft_seat", "frontier_reactive_double_displacement",
]


def build_scenario_dict(case_id: str, direction: str, offset: list[float], timing_bucket: str) -> dict:
    step = 15 if timing_bucket == "early" else (25 if timing_bucket == "middle" else 35)
    prox = 0.10 if timing_bucket == "early" else (0.08 if timing_bucket == "middle" else 0.05)

    return {
        "scenario_id": f"scen_{case_id}",
        "change_family": "SPATIAL_DISPLACEMENT",
        "change_type": "object_displacement",
        "trigger": {
            "trigger_type": "pre_grasp_proximity",
            "step": step,
            "proximity_threshold_m": prox,
            "force_threshold_n": None,
        },
        "intervention": {
            "direction": direction,
            "offset_m": offset,
            "impulse_velocity_mps": None,
        },
        "safety_thresholds": {
            "max_safe_force_n": 150.0,
            "enforce_torque_limits": True,
        },
    }


def main() -> None:
    # 1. Gather all 240 tasks (210 core tasks + 30 Domain 8 frontier tasks)
    all_task_entries = []
    
    # Core tasks (1..210)
    for t_id, spec in TASK_MAP.items():
        all_task_entries.append((spec.number, t_id, spec.domain))

    # Frontier tasks (211..240)
    for t in DOMAIN_8_FRONTIER_TASKS:
        all_task_entries.append((t.task_number, t.task_id, t.domain))

    all_task_entries.sort(key=lambda x: x[0])

    print(f"Total tasks gathered: {len(all_task_entries)}")

    # 2. Build Full Manifest (4 matched pairs per task = 960 matched pairs, 1,920 rollouts)
    full_cases = []
    timing_buckets = ["early", "middle", "late", "middle"]
    dir_keys = ["+x", "-x", "+y", "-y"]

    case_counter = 0
    for task_num, task_id, domain_raw in all_task_entries:
        d_id, d_name = DOMAIN_MAPPING.get(domain_raw, ("D1", domain_raw))
        
        for p_idx in range(4):
            dir_k = dir_keys[p_idx % len(dir_keys)]
            t_bucket = timing_buckets[p_idx % len(timing_buckets)]
            seed = 1000 + task_num * 10 + p_idx
            case_id = f"afford_{d_id.lower()}_{task_num:03d}_p{p_idx}_{dir_k}"
            offset = DIRECTIONS[dir_k]

            scenario = build_scenario_dict(case_id, dir_k, offset, t_bucket)

            full_cases.append({
                "case_id": case_id,
                "domain_id": d_id,
                "domain_name": d_name,
                "task_id": task_id,
                "task_number": task_num,
                "policy_seed": seed,
                "timing_bucket": t_bucket,
                "scenario": scenario,
            })
            case_counter += 1

    full_manifest = {
        "benchmark_id": "AFFORD-Bench",
        "benchmark_version": "1.0.0",
        "track": "full",
        "protocol": {
            "arms": ["control", "intervention"],
            "scoring_track": "compliant_dynamic_completion",
            "safety_constraints": {
                "max_contact_force_n": 150.0,
                "joint_torque_limits_nm": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
            },
        },
        "cases": full_cases,
    }

    full_path = MANIFESTS_DIR / "afford_bench_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"Wrote Full Manifest: {full_path} ({len(full_cases)} cases)")

    # 3. Build Lite Manifest (24 tasks * 4 matched pairs = 96 cases, 192 rollouts)
    lite_task_set = set(LITE_TASKS)
    lite_cases = [c for c in full_cases if c["task_id"] in lite_task_set]
    lite_cases.sort(key=lambda x: (x["task_number"], x["case_id"]))

    lite_manifest = {
        "benchmark_id": "AFFORD-Bench",
        "benchmark_version": "1.0.0",
        "track": "lite",
        "protocol": {
            "arms": ["control", "intervention"],
            "scoring_track": "compliant_dynamic_completion",
            "safety_constraints": {
                "max_contact_force_n": 150.0,
                "joint_torque_limits_nm": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
            },
        },
        "cases": lite_cases,
    }

    lite_path = LITE_DIR / "afford_bench_lite.json"
    with open(lite_path, "w", encoding="utf-8") as f:
        json.dump(lite_manifest, f, indent=2)
    print(f"Wrote Lite Manifest: {lite_path} ({len(lite_cases)} cases)")

    # 4. Write CSV index for Lite
    csv_path = LITE_DIR / "case_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["case_id", "domain_id", "task_number", "task_id", "policy_seed", "timing_bucket", "direction"])
        for c in lite_cases:
            writer.writerow([
                c["case_id"],
                c["domain_id"],
                c["task_number"],
                c["task_id"],
                c["policy_seed"],
                c["timing_bucket"],
                c["scenario"]["intervention"]["direction"],
            ])
    print(f"Wrote Lite CSV Index: {csv_path}")

    # 5. Write Lite README
    readme_path = LITE_DIR / "README.md"
    readme_content = f"""# AFFORD-Bench Lite Benchmark Track

**AFFORD-Bench Lite** is a frozen, deterministic subset of AFFORD-Bench designed for rapid policy iteration, hyperparameter tuning, and CI testing.

## Specification

- **Total Matched Pairs**: {len(lite_cases)}
- **Total Scored Rollouts**: {len(lite_cases) * 2} (96 Base Control + 96 Dynamic Intervention)
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

1. **Exact Pre-Event Replay**: Both Base and Dynamic episodes share identical random seeds, initial scene states, and executed pre-event action prefixes $a_{{0:t^*}}$.
2. **Force & Torque Verification**:
   - $F_{{\\max}} \\le 150\\,\\text{{N}}$ (ISO/TS 15066 safety threshold).
   - $\\boldsymbol{{\\tau}}_{{\\text{{act}}}} \\le [87, 87, 87, 87, 12, 12, 12]\\,\\text{{N}}\\cdot\\text{{m}}$ (Panda limits).
3. **Scoring**:
   - Headline metric: **Compliant Dynamic Success Rate** ($S_{{\\text{{safe}}}} = S_{{\\text{{dyn}}}} \\land \\neg \\text{{safety\\_violation}}$).

## Running Lite Evaluation

```bash
python scripts/evaluate_paired_dynamic.py --tasks representative --max-cases 96
```
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Wrote Lite README: {readme_path}")


if __name__ == "__main__":
    main()
