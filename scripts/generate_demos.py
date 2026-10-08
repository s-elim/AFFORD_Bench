"""CLI Script to batch generate Robomimic HDF5 demonstration datasets using 100% local oracles."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Ensure afford_pro is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer, make_env_args


def main():
    available_tasks = afford_pro.list_tasks()
    parser = argparse.ArgumentParser(description="Generate demonstration datasets using 100% local oracles.")
    parser.add_argument("--task", type=str, default="industrial_shaft_assembly", choices=available_tasks, help="Task ID")
    parser.add_argument("--num-demos", type=int, default=5, help="Number of successful demonstration episodes to collect")
    parser.add_argument("--perturbation", type=str, default="nominal", choices=[
        "nominal", "p1_object", "p2_spatial", "p3_visual", "p4_distractor", "combined_ood"
    ])
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", action="store_true", help="Also record metric depth and camera parameters")
    args = parser.parse_args()

    entry = afford_pro.get_task_entry(args.task)
    env = afford_pro.make(args.task, perturbation_mode=args.perturbation, render_depth=args.depth)
    oracle = afford_pro.get_oracle(args.task, env)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / f"{args.task}_{args.perturbation}.hdf5"

    print(f"\n================================================================================")
    print(f" AFFORD-PRO-150: DEMONSTRATION GENERATOR (100% LOCAL / ZERO API COST)")
    print(f"================================================================================")
    print(f" Domain       : {entry.domain}")
    print(f" Task         : {args.task} (Task #{entry.task_number:03d})")
    print(f" Instruction  : {entry.instruction}")
    print(f" Target Demos : {args.num_demos}")
    print(f" Perturbation : {args.perturbation}")
    print(f" Output File  : {hdf5_path}")
    print(f"================================================================================\n")

    model_xml = env.get_model_xml() if hasattr(env, "get_model_xml") else None
    writer = RobomimicHDF5Writer(
        file_path=hdf5_path,
        env_name=args.task,
        task_id=args.task,
        instruction=entry.instruction,
        env_args=make_env_args(args.task, perturbation_mode=args.perturbation, render_depth=args.depth),
        model_file=model_xml,
    )

    if args.depth:
        env.reset(seed=args.seed)
        for key, value in env.camera_parameters().items():
            writer.data_grp.attrs[f"camera_{key}"] = value

    t0 = time.monotonic()
    collected = 0
    total_steps = 0

    for ep in range(args.num_demos):
        ep_seed = args.seed + ep
        ep_t0 = time.monotonic()

        actions, observations, rewards, dones = oracle.generate_episode(seed=ep_seed)
        success = env.check_success()

        if success:
            demo_key = writer.add_demonstration(actions, observations, rewards, dones,
                                                states=oracle.last_states, seed=ep_seed)
            collected += 1
            total_steps += len(actions)
            dur = time.monotonic() - ep_t0
            fps = len(actions) / max(0.001, dur)
            print(f" [{collected}/{args.num_demos}] {demo_key} collected! Steps: {len(actions):3d} | Time: {dur:.2f}s ({fps:.0f} FPS)")
        else:
            print(f" [RETRY] Seed {ep_seed} did not satisfy success condition; retrying...")

    writer.close()
    total_time = time.monotonic() - t0
    overall_fps = total_steps / max(0.001, total_time)

    print(f"\n================================================================================")
    print(f" DATASET GENERATION COMPLETE")
    print(f" Total Demos     : {collected}")
    print(f" Total Timesteps : {total_steps}")
    print(f" Wall Time       : {total_time:.2f}s")
    print(f" Throughput      : {overall_fps:.1f} FPS")
    print(f" Dataset Saved   : {hdf5_path}")
    print(f" API Credits Used: $0.00 (Zero API calls)")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
