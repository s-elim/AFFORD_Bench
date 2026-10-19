"""Master Benchmark Suite Runner for AFFORD-PRO-150.

Evaluates policies across all registered tasks and OOD perturbation splits,
and generates a publication-ready Markdown results table.
"""
# Deprecated legacy tool: runs the single-body legacy task set (task_set="legacy") to reproduce its evidence.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro


def main():
    parser = argparse.ArgumentParser(description="Run complete AFFORD-PRO-150 benchmark suite.")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "industrial", "kitchen", "tabletop"])
    parser.add_argument("--policy", type=str, default="oracle", choices=["oracle", "random"])
    parser.add_argument("--num-episodes", type=int, default=3, help="Episodes per split")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    all_task_ids = afford_pro.list_tasks("legacy")
    selected_tasks = []
    for tid in all_task_ids:
        entry = afford_pro.get_task_entry(tid, "legacy")
        if args.domain == "all":
            selected_tasks.append(tid)
        elif args.domain == "industrial" and "Industrial" in entry.domain:
            selected_tasks.append(tid)
        elif args.domain == "kitchen" and "Culinary" in entry.domain:
            selected_tasks.append(tid)
        elif args.domain == "tabletop" and "Tabletop" in entry.domain:
            selected_tasks.append(tid)

    splits = ["nominal", "p1_object", "p2_spatial", "p3_visual", "combined_ood"]

    print("\n================================================================================")
    print(f" AFFORD-PRO-150: COMPREHENSIVE BENCHMARK EVALUATION SUITE")
    print(f" Policy Under Test : {args.policy.upper()}")
    print(f" Total Tasks       : {len(selected_tasks)}")
    print(f" Episodes Per Split: {args.num_episodes}")
    print(f" Target Venues     : CoRL 2026 / RSS 2026 / NeurIPS 2026")
    print("================================================================================\n")

    suite_results = []
    start_suite = time.monotonic()

    for idx, tid in enumerate(selected_tasks, start=1):
        entry = afford_pro.get_task_entry(tid, "legacy")
        print(f"[{idx}/{len(selected_tasks)}] Evaluating Task #{entry.task_number:03d}: {tid} ({entry.domain})...")

        task_scores = {"task_id": tid, "domain": entry.domain, "number": entry.task_number}
        for split in splits:
            env = afford_pro.make(tid, perturbation_mode=split, task_set="legacy")
            oracle = afford_pro.get_oracle(tid, env)

            successes = []
            for ep in range(args.num_episodes):
                ep_seed = args.seed + ep
                if args.policy == "oracle":
                    oracle.generate_episode(seed=ep_seed)
                    successes.append(float(env.check_success()))
                elif args.policy == "random":
                    env.reset(seed=ep_seed)
                    done = False
                    while not done and env.current_step < 60:
                        act = env.action_space.sample()
                        obs, r, term, trunc, info = env.step(act)
                        done = term or trunc
                    successes.append(float(env.check_success()))

            sr = float(np.mean(successes)) * 100.0
            task_scores[split] = sr

        # Compute GDI for this task
        id_sr = task_scores["nominal"]
        ood_avg = np.mean([task_scores[s] for s in splits if s != "nominal"])
        gdi = ((id_sr - ood_avg) / max(1e-4, id_sr)) * 100.0 if id_sr > 0 else 0.0
        task_scores["ood_avg"] = ood_avg
        task_scores["gdi"] = gdi
        suite_results.append(task_scores)

    total_time = time.monotonic() - start_suite

    # Print Publication-Grade Markdown Results Table
    print("\n" + "=" * 80)
    print(" PUBLICATION BENCHMARK RESULTS TABLE (Markdown)")
    print("=" * 80 + "\n")

    md_table = []
    md_table.append("| Task ID | Domain | Nominal (ID) | P1 (Object) | P2 (Spatial) | P3 (Visual) | Combined OOD | OOD Avg | GDI (%) ↓ |")
    md_table.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in suite_results:
        md_table.append(
            f"| `{r['task_id']}` | {r['domain']} | {r['nominal']:.1f}% | {r['p1_object']:.1f}% | {r['p2_spatial']:.1f}% | {r['p3_visual']:.1f}% | {r['combined_ood']:.1f}% | {r['ood_avg']:.1f}% | {r['gdi']:.1f}% |"
        )

    # Compute Macro Averages
    macro_id = np.mean([r["nominal"] for r in suite_results])
    macro_p1 = np.mean([r["p1_object"] for r in suite_results])
    macro_p2 = np.mean([r["p2_spatial"] for r in suite_results])
    macro_p3 = np.mean([r["p3_visual"] for r in suite_results])
    macro_ood = np.mean([r["combined_ood"] for r in suite_results])
    macro_ood_avg = np.mean([r["ood_avg"] for r in suite_results])
    macro_gdi = ((macro_id - macro_ood_avg) / max(1e-4, macro_id)) * 100.0 if macro_id > 0 else 0.0

    md_table.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md_table.append(
        f"| **MACRO AVERAGE** | **All Evaluated Tasks** | **{macro_id:.1f}%** | **{macro_p1:.1f}%** | **{macro_p2:.1f}%** | **{macro_p3:.1f}%** | **{macro_ood:.1f}%** | **{macro_ood_avg:.1f}%** | **{macro_gdi:.1f}%** |"
    )

    table_str = "\n".join(md_table)
    print(table_str)

    print("\n" + "=" * 80)
    print(f" EVALUATION FINISHED IN {total_time:.2f}s (Total Tasks: {len(selected_tasks)})")
    print(f" Zero API Cost ($0.00) | Full Local MuJoCo C++/Python Execution")
    print("=" * 80 + "\n")

    # Save to file
    out_table_path = ROOT / "BENCHMARK_RESULTS.md"
    with open(out_table_path, "w", encoding="utf-8") as f:
        f.write("# AFFORD-PRO-150 Benchmark Evaluation Report\n\n")
        f.write(f"- **Evaluated Policy**: `{args.policy.upper()}`\n")
        f.write(f"- **Total Evaluated Tasks**: `{len(selected_tasks)}`\n")
        f.write(f"- **Episodes per Split**: `{args.num_episodes}`\n")
        f.write(f"- **Execution Duration**: `{total_time:.2f}s`\n\n")
        f.write("## Out-of-Distribution Generalization Performance\n\n")
        f.write(table_str)
        f.write("\n\n*Generalization Degradation Index (GDI): $\\text{GDI} = \\frac{\\text{ID}_{\\text{SR}} - \\text{OOD}_{\\text{SR}}}{\\text{ID}_{\\text{SR}}} \\times 100\\%$ (lower is better)*\n")

    print(f"Results saved to: {out_table_path}")


if __name__ == "__main__":
    main()
