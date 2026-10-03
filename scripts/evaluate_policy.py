"""Policy Evaluation Harness for AFFORD-PRO-150 across In-Distribution and OOD Perturbations."""
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


def run_evaluation(
    task: str,
    perturbation: str,
    num_episodes: int = 5,
    policy_type: str = "oracle",
    seed: int = 1000,
) -> dict:
    env = afford_pro.make(task, perturbation_mode=perturbation)
    oracle = afford_pro.get_oracle(task, env)

    successes = []
    episode_lengths = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        if policy_type == "oracle":
            oracle.generate_episode(seed=ep_seed)
            success = env.check_success()
            ep_len = env.current_step
        elif policy_type == "random":
            env.reset(seed=ep_seed)
            done = False
            while not done and env.current_step < 100:
                act = env.action_space.sample()
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
            success = env.check_success()
            ep_len = env.current_step
        else:
            raise NotImplementedError(f"Custom policy loader for '{policy_type}' can be registered here.")

        successes.append(float(success))
        episode_lengths.append(ep_len)

    sr = float(np.mean(successes))
    avg_len = float(np.mean(episode_lengths))
    return {"sr": sr, "avg_len": avg_len, "num_episodes": num_episodes}


def main():
    available_tasks = afford_pro.list_tasks()
    parser = argparse.ArgumentParser(description="Evaluate VLA models & policies on AFFORD-PRO-150.")
    parser.add_argument("--task", type=str, default="industrial_shaft_assembly", choices=available_tasks)
    parser.add_argument("--policy", type=str, default="oracle", choices=["oracle", "random"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    entry = afford_pro.get_task_entry(args.task)
    splits = ["nominal", "p1_object", "p2_spatial", "p3_visual", "combined_ood"]

    print("\n================================================================================")
    print(f" AFFORD-PRO-150: OOD GENERALIZATION BENCHMARK EVALUATOR")
    print(f" Domain       : {entry.domain}")
    print(f" Task         : {args.task} (Task #{entry.task_number:03d})")
    print(f" Policy       : {args.policy.upper()}")
    print(f" Instruction  : {entry.instruction}")
    print(f" Episodes/Split: {args.num_episodes}")
    print("================================================================================")

    results = {}
    for split in splits:
        t0 = time.monotonic()
        res = run_evaluation(args.task, split, num_episodes=args.num_episodes, policy_type=args.policy, seed=args.seed)
        dur = time.monotonic() - t0
        results[split] = res["sr"]
        print(f" Split: {split:<15} | Success Rate: {res['sr']*100:5.1f}% | Avg Steps: {res['avg_len']:4.1f} | Time: {dur:.1f}s")

    id_sr = results["nominal"]
    ood_avg = np.mean([results[s] for s in splits if s != "nominal"])
    gdi = ((id_sr - ood_avg) / max(1e-4, id_sr)) * 100 if id_sr > 0 else 0.0

    print("================================================================================")
    print(f" BENCHMARK SUMMARY")
    print(f" In-Distribution (ID) Success Rate   : {id_sr * 100:.1f}%")
    print(f" Out-of-Distribution (OOD) Avg SR    : {ood_avg * 100:.1f}%")
    print(f" Generalization Degradation Index (GDI): {gdi:.1f}% (Lower is better)")
    print(f" API Credits Used                    : $0.00 (Zero API calls)")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
