"""Evaluate policies or oracles on AFFORD-Bench Paired Base/Dynamic Suites.

Executes matched counterfactual pairs (Base control vs. Dynamic intervention)
with force & torque safety verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.dynamic_suite import (
    DISPLACEMENT_DIRECTIONS,
    REPRESENTATIVE_D1_D7_TASKS,
    build_paired_dynamic_suite,
    run_paired_case,
    summarize_paired_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paired Base/Dynamic robustness.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="representative",
        help="Comma-separated task IDs, 'representative', or 'all'.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,101,2024",
        help="Comma-separated random seeds, e.g. '42,101,2024'.",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="+x,-x,+y,-y",
        help="Displacement directions (+x, -x, +y, -y, +xy).",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="pre_grasp_proximity",
        choices=["pre_grasp_proximity", "mid_approach_step"],
        help="Trigger mechanism for object displacement.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of paired cases to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paired_dynamic",
        help="Directory to write output summary and logs.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render frames (slower, default runs headless).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks == "representative":
        task_ids = REPRESENTATIVE_D1_D7_TASKS
    elif args.tasks == "all":
        task_ids = [t.task_id for t in afford_pro.list_all_tasks()]
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    cases = build_paired_dynamic_suite(
        task_ids=task_ids,
        seeds=seeds,
        directions=directions,
        trigger_type=args.trigger,
    )

    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    print("=" * 80)
    print(f"AFFORD-Bench Paired Base/Dynamic Evaluation Suite")
    print(f"Total Paired Cases: {len(cases)} ({len(cases)*2} total rollouts)")
    print(f"Tasks: {len(task_ids)} | Seeds: {len(seeds)} | Directions: {directions}")
    print(f"Trigger: {args.trigger}")
    print("=" * 80)

    outcomes = []
    jsonl_path = args.output_dir / "paired_outcomes.jsonl"

    t0 = time.time()
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for idx, case in enumerate(cases):
            outcome = run_paired_case(case, policy_fn=None, use_headless=not args.render)
            outcomes.append(outcome)
            f_jsonl.write(json.dumps(outcome.to_dict()) + "\n")
            f_jsonl.flush()

            status = f"[{idx+1}/{len(cases)}] {case.task_id} ({case.displacement_direction}) -> {outcome.category.upper()}"
            force_str = f"| Peak F: Base={outcome.base_peak_force:.1f}N, Dyn={outcome.dynamic_peak_force:.1f}N"
            safety_str = " | SAFE" if not outcome.dynamic_safety_violation else " | SAFETY VIOLATION!"
            print(f"{status:<65} {force_str:<32} {safety_str}")

    elapsed = time.time() - t0
    summary = summarize_paired_outcomes(outcomes)
    summary["elapsed_seconds"] = elapsed
    summary["rollouts_per_second"] = (len(cases) * 2) / max(elapsed, 1e-3)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"Base Success Rate:               {summary['base_success_rate']*100:.1f}%")
    print(f"Dynamic Success Rate:            {summary['dynamic_success_rate']*100:.1f}%")
    print(f"Paired Performance Drop:         {summary['paired_gap_pp']:.1f} percentage points")
    print(f"Conditional Regression Rate:     {summary['conditional_regression_rate']*100:.1f}%")
    print("-" * 80)
    print("Outcome Breakdown:")
    for k, v in summary["percentages"].items():
        print(f"  {k:<28}: {v:.1f}% ({summary['outcomes'][k]} pairs)")
    print("-" * 80)
    print("Force & Contact Sensing Telemetry (AFFORD-Bench Advantage):")
    fs = summary["force_and_safety"]
    print(f"  Dynamic Force Violations (>150N):  {fs['dynamic_force_violation_count']} ({fs['dynamic_force_violation_rate']*100:.1f}%)")
    print(f"  Dynamic Torque Saturations:        {fs['dynamic_torque_saturation_count']} ({fs['dynamic_torque_saturation_rate']*100:.1f}%)")
    print(f"  Compliant Dynamic Success Rate:    {fs['compliant_dynamic_success_rate']*100:.1f}%")
    print(f"  Max Dynamic Contact Force:         {fs['maximum_dynamic_force_recorded_n']:.2f} N")
    print(f"Detailed logs saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
