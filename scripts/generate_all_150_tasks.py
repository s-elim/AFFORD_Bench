"""Batch Demonstration Dataset Generator for the Entire AFFORD-PRO-150 Benchmark Suite.

Generates 50 expert demonstrations per task across all 150 tasks in standard Robomimic HDF5 format.
Runs 100% locally in MuJoCo C++/Python (Zero external API calls, $0.00 cost).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer, make_env_args


def is_task_already_done(hdf5_path: Path, target_demos: int) -> bool:
    """Check if the dataset already contains at least target_demos demonstrations."""
    if not hdf5_path.exists():
        return False
    try:
        with h5py.File(str(hdf5_path), "r") as f:
            if "data" in f:
                keys = [k for k in f["data"].keys() if k.startswith("demo_")]
                return len(keys) >= target_demos
    except Exception:
        return False
    return False


def generate_task_demos(
    task_id: str,
    num_demos: int = 50,
    perturbation: str = "nominal",
    output_dir: Path = ROOT / "datasets",
    base_seed: int = 42,
    depth: bool = False,
) -> tuple[int, int, float]:
    """Generate demonstrations for a single task."""
    entry = afford_pro.get_task_entry(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / f"{task_id}_{perturbation}.hdf5"

    if is_task_already_done(hdf5_path, num_demos):
        print(f"  [SKIP] Task #{entry.task_number:03d} '{task_id}' already has {num_demos} demos.", flush=True)
        return num_demos, 0, 0.0

    env = afford_pro.make(task_id, perturbation_mode=perturbation, render_depth=depth)
    oracle = afford_pro.get_oracle(task_id, env)
    model_xml = env.get_model_xml() if hasattr(env, "get_model_xml") else None

    writer = RobomimicHDF5Writer(
        file_path=hdf5_path,
        env_name=task_id,
        task_id=task_id,
        instruction=entry.instruction,
        env_args=make_env_args(task_id, perturbation_mode=perturbation, render_depth=depth),
        model_file=model_xml,
    )
    if depth:
        env.reset(seed=base_seed)
        for key, value in env.camera_parameters().items():
            writer.data_grp.attrs[f"camera_{key}"] = value

    t0 = time.monotonic()
    collected = 0
    total_steps = 0
    attempt = 0

    while collected < num_demos and attempt < num_demos * 3:
        ep_seed = base_seed + attempt
        attempt += 1

        actions, observations, rewards, dones = oracle.generate_episode(seed=ep_seed)
        if env.check_success():
            writer.add_demonstration(actions, observations, rewards, dones, states=oracle.last_states, seed=ep_seed)
            collected += 1
            total_steps += len(actions)
            if collected % 5 == 0 or collected == num_demos:
                elapsed = time.monotonic() - t0
                fps = total_steps / max(0.001, elapsed)
                print(f"    -> [{collected:2d}/{num_demos}] Demos collected ({total_steps:5d} steps, {fps:.0f} FPS)", flush=True)

    writer.close()
    duration = time.monotonic() - t0
    return collected, total_steps, duration


def main():
    parser = argparse.ArgumentParser(description="Batch generate 50 demonstrations per task across all 210 tasks.")
    parser.add_argument("--num-demos", type=int, default=50, help="Number of demonstrations per task")
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "industrial", "kitchen", "tabletop", "lab", "long_horizon", "chain_precision", "chain_service"],
    )
    parser.add_argument("--start-task", type=int, default=1, help="First task number (1-210)")
    parser.add_argument("--end-task", type=int, default=210, help="Last task number (1-210)")
    parser.add_argument("--perturbation", type=str, default="nominal", choices=["nominal", "p1_object", "p2_spatial", "p3_visual", "combined_ood"])
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", action="store_true",
                        help="Also record float32 metric depth for both cameras and their camera parameters (much larger files)")
    args = parser.parse_args()

    all_task_ids = afford_pro.list_tasks()
    selected_tasks = []

    domain_match = {
        "industrial": "Industrial",
        "kitchen": "Culinary",
        "tabletop": "Tabletop",
        "lab": "Laboratory",
        "long_horizon": "Long-Horizon",
        "chain_precision": "Precision Chaining",
        "chain_service": "Service & Culinary",
    }

    for tid in all_task_ids:
        entry = afford_pro.get_task_entry(tid)
        if args.start_task <= entry.task_number <= args.end_task:
            if args.domain == "all":
                selected_tasks.append(tid)
            elif domain_match[args.domain] in entry.domain:
                selected_tasks.append(tid)

    out_path = ROOT / args.output_dir

    print("\n" + "=" * 80)
    print(f" AFFORD-Bench: MASSIVE MULTI-TASK DEMONSTRATION GENERATOR (210 TASKS)")
    print(f" Target Tasks      : {len(selected_tasks)} tasks (Task #{args.start_task} to #{args.end_task})")
    print(f" Demos Per Task    : {args.num_demos}")
    print(f" Total Target Demos: {len(selected_tasks) * args.num_demos:,} demonstrations")
    print(f" Perturbation Split: {args.perturbation}")
    print(f" Output Directory  : {out_path}")
    print(f" API Cost          : $0.00 (Zero external API calls, 100% local MuJoCo)")
    print("=" * 80 + "\n")

    grand_total_demos = 0
    grand_total_steps = 0
    suite_start = time.monotonic()

    for idx, tid in enumerate(selected_tasks, start=1):
        entry = afford_pro.get_task_entry(tid)
        print(f"[{idx}/{len(selected_tasks)}] Processing Task #{entry.task_number:03d}: '{tid}' [{entry.domain}]...")

        c, s, d = generate_task_demos(
            task_id=tid,
            num_demos=args.num_demos,
            perturbation=args.perturbation,
            output_dir=out_path,
            base_seed=args.seed + entry.task_number * 1000,
            depth=args.depth,
        )
        grand_total_demos += c
        grand_total_steps += s

    total_time = time.monotonic() - suite_start
    overall_fps = grand_total_steps / max(0.001, total_time)

    print("\n" + "=" * 80)
    print(f" AFFORD-PRO-150 BATCH GENERATION SUMMARY")
    print(f" Total Completed Tasks : {len(selected_tasks)}")
    print(f" Total Demos Generated : {grand_total_demos:,}")
    print(f" Total Physical Steps  : {grand_total_steps:,}")
    print(f" Total Elapsed Time    : {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f" Effective Throughput  : {overall_fps:.1f} FPS")
    print(f" External API Cost     : $0.00")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
