"""Export fixed initial simulator states per task across AFFORD-Bench.

Similar to LIBERO's fixed initial states, this eliminates RNG variance during benchmark
evaluation: policies are initialized directly into pre-settled physical states.

Output format per task: data/init_states/{task_id}_init_states.npz containing:
  - 'states': (N, S) float64 array of MuJoCo physics states + gripper command
  - 'seeds': (N,) int64 array of reset seeds used to generate the initial states
  - 'task_id': str
"""
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


def export_task_initial_states(
    task_id: str,
    num_states: int = 50,
    output_dir: Path = ROOT / "data" / "init_states",
    base_seed: int = 200_000,
) -> Path:
    """Generate and save num_states settled initial simulator states for task_id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{task_id}_init_states.npz"

    entry = afford_pro.get_task_entry(task_id)
    task_base_seed = base_seed + entry.task_number * 1000

    env = afford_pro.make(task_id, perturbation_mode="nominal")
    states = []
    seeds = []

    for idx in range(num_states):
        ep_seed = task_base_seed + idx
        env.reset(seed=ep_seed)
        state = env.get_sim_state()
        states.append(state)
        seeds.append(ep_seed)

    env.close()

    states_arr = np.asarray(states, dtype=np.float64)
    seeds_arr = np.asarray(seeds, dtype=np.int64)

    np.savez_compressed(
        out_file,
        states=states_arr,
        seeds=seeds_arr,
        task_id=task_id,
        num_states=num_states,
    )
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Export fixed initial simulator states for AFFORD-Bench tasks.")
    parser.add_argument("--domain", type=str, default="all",
                        choices=["all", "industrial", "kitchen", "tabletop", "lab", "long_horizon", "chain_precision", "chain_service"])
    parser.add_argument("--task", type=str, default=None, help="Export for a single task ID")
    parser.add_argument("--num-states", type=int, default=50, help="Number of fixed initial states per task")
    parser.add_argument("--output-dir", type=str, default="data/init_states")
    parser.add_argument("--base-seed", type=int, default=200_000)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    all_tasks = afford_pro.list_tasks()

    if args.task:
        selected_tasks = [args.task]
    elif args.domain == "all":
        selected_tasks = all_tasks
    else:
        domain_match = {
            "industrial": "Industrial",
            "kitchen": "Culinary",
            "tabletop": "Tabletop",
            "lab": "Laboratory",
            "long_horizon": "Long-Horizon",
            "chain_precision": "Precision Chaining",
            "chain_service": "Service & Culinary",
        }
        target_dom = domain_match[args.domain]
        selected_tasks = [t for t in all_tasks if target_dom in afford_pro.get_task_entry(t).domain]

    print("\n" + "=" * 80)
    print(f" AFFORD-Bench: INITIAL SIMULATOR STATE EXPORTER")
    print(f" Target Tasks       : {len(selected_tasks)} tasks")
    print(f" States Per Task    : {args.num_states}")
    print(f" Output Directory   : {out_dir}")
    print(f" Base Seed Offset   : {args.base_seed}")
    print("=" * 80 + "\n")

    t0 = time.monotonic()
    for idx, tid in enumerate(selected_tasks, 1):
        t_start = time.monotonic()
        path = export_task_initial_states(tid, num_states=args.num_states, output_dir=out_dir, base_seed=args.base_seed)
        dur = time.monotonic() - t_start
        print(f"[{idx:3d}/{len(selected_tasks):3d}] Exported '{tid}' -> {path.name} ({dur:.2f}s)", flush=True)

    total_time = time.monotonic() - t0
    print(f"\nCompleted exporting initial states in {total_time:.2f}s.\n")


if __name__ == "__main__":
    main()
