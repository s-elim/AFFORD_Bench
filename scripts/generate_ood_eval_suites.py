"""Out-of-Distribution (OOD) Evaluation Suite Generator for AFFORD-Bench.

Generates standardized perturbed evaluation demonstration suites for offline benchmark
evaluation, replay testing, and multi-modal distribution shift benchmarking.

Usage Examples:
  # Generate Spatial OOD evaluation suite for Industrial domain
  python scripts/generate_ood_eval_suites.py --domain industrial --perturbation p2_spatial --num-demos 10

  # Generate Physical-Dynamic OOD evaluation suite for Kitchen domain
  python scripts/generate_ood_eval_suites.py --domain kitchen --perturbation p1_object --num-demos 10

  # Generate Combined Multi-Axis OOD evaluation suite across all domains
  python scripts/generate_ood_eval_suites.py --domain all --perturbation combined_ood --num-demos 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from scripts.generate_all_150_tasks import generate_task_demos


def main():
    parser = argparse.ArgumentParser(description="Generate OOD evaluation suites for AFFORD-Bench.")
    parser.add_argument("--domain", type=str, default="industrial", choices=["all", "industrial", "kitchen", "tabletop", "lab", "long_horizon"])
    parser.add_argument(
        "--perturbation",
        type=str,
        default="combined_ood",
        choices=["all", "p1_object", "p2_spatial", "p3_visual", "combined_ood"],
    )
    parser.add_argument("--num-demos", type=int, default=10, help="Number of evaluation demonstrations per task")
    parser.add_argument("--output-dir", type=str, default="datasets/eval_suites")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    all_task_ids = afford_pro.list_tasks()
    selected_tasks = []

    domain_match = {
        "industrial": "Industrial",
        "kitchen": "Culinary",
        "tabletop": "Tabletop",
        "lab": "Laboratory",
        "long_horizon": "Long-Horizon",
    }

    for tid in all_task_ids:
        entry = afford_pro.get_task_entry(tid)
        if args.domain == "all":
            selected_tasks.append(tid)
        elif domain_match[args.domain] in entry.domain:
            selected_tasks.append(tid)

    perturbation_list = (
        ["p1_object", "p2_spatial", "p3_visual", "combined_ood"]
        if args.perturbation == "all"
        else [args.perturbation]
    )

    out_base = ROOT / args.output_dir

    print("\n" + "=" * 80, flush=True)
    print(" AFFORD-Bench: OOD EVALUATION SUITE GENERATOR", flush=True)
    print(f" Target Domain       : {args.domain.upper()}", flush=True)
    print(f" Tasks In Scope      : {len(selected_tasks)} tasks", flush=True)
    print(f" Perturbation Splits : {', '.join(perturbation_list)}", flush=True)
    print(f" Demos Per Task/Split: {args.num_demos}", flush=True)
    print(f" Output Directory    : {out_base}", flush=True)
    print("=" * 80 + "\n", flush=True)

    t0 = time.monotonic()
    total_generated = 0
    total_steps = 0

    for p_split in perturbation_list:
        split_dir = out_base / p_split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating OOD Suite: '{p_split.upper()}' ---", flush=True)

        for idx, tid in enumerate(selected_tasks, 1):
            entry = afford_pro.get_task_entry(tid)
            print(f"[{idx:2d}/{len(selected_tasks):2d}] Task #{entry.task_number:03d}: '{tid}' [{entry.domain}]...", flush=True)
            c, s, d = generate_task_demos(
                task_id=tid,
                num_demos=args.num_demos,
                perturbation=p_split,
                output_dir=split_dir,
                base_seed=args.seed + entry.task_number * 1000,
            )
            total_generated += c
            total_steps += s

    elapsed = time.monotonic() - t0
    print("\n" + "=" * 80, flush=True)
    print(" OOD EVALUATION SUITE GENERATION COMPLETE", flush=True)
    print(f" Total Demos Generated : {total_generated:,}", flush=True)
    print(f" Total Physical Steps  : {total_steps:,}", flush=True)
    print(f" Total Elapsed Time    : {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f" Output Location       : {out_base}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
