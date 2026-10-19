"""Runtime validation of the AFFORD-Bench tasks against the T9 acceptance gate (docs/roadmap/ROADMAP.md).

For each task: build the environment (every goal and pre-placed start is dropped into place and must come to rest),
reset it --resets times (start variability of every moved body, start fallbacks, goals already true at reset), and run
the scripted oracle for --episodes episodes (success within the task's step budget, steps, goals reached).

    python -I scripts/validate_bench_tasks.py --episodes 10 --resets 10 --workers 8 \\
        --out docs/roadmap/evidence/task_validation_<date>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED_BLOCK = 600_000   # validation seeds: SEED_BLOCK + 1000 * task_number + j
AUDIT_BLOCK = 700_000  # start-layout audit seeds
FLOOR_BLOCK = 800_000  # random- and null-policy seeds


def _audit_starts(env, number: int, resets: int) -> dict:
    moved = env.task_spec.bodies_moved()
    starts = {b: [] for b in moved}
    fallbacks, true_at_reset = 0, 0
    for j in range(resets):
        env.reset(seed=AUDIT_BLOCK + 1000 * number + j)
        fallbacks += int(env.start_fallback)
        for b in moved:
            starts[b].append(env.data.xpos[env.body_ids[b]][:2].copy())
        true_at_reset += int(any(env._goal_satisfied_at_start(i, g) for i, g in enumerate(env.task_spec.goals)
                                 if not env._predecessors(i)))
    spread = {b: float(np.linalg.norm(np.std(np.array(v), axis=0))) for b, v in starts.items()}
    return {"start_spread_m": spread, "start_fallbacks": fallbacks, "goal_true_at_reset": true_at_reset}


def _floors(env, task_id: str, suite: str, number: int, episodes: int) -> dict:
    """Successes of the random and null policies (must be zero: no task is solved by chance or by doing nothing)."""
    from afford_pro.evaluation.rollout import NullPolicy, RandomPolicy, run_policy_episode

    out = {}
    for policy in (RandomPolicy(), NullPolicy()):
        runs = [run_policy_episode(env, policy, task_id, suite, FLOOR_BLOCK + 1000 * number + j, env.max_episode_steps)
                for j in range(episodes)]
        out[policy.name] = sum(r.success for r in runs)
    return out


def validate(args: tuple[int, int, int, str, int]) -> dict:
    number, episodes, resets, suite, floor_episodes = args
    from afford_pro.envs.headless import headless
    from afford_pro.bench.env import GoalFrankaEnv
    from afford_pro.bench.oracle import GoalOracle
    from afford_pro.bench.tasks import TASK_BY_NUMBER

    spec = TASK_BY_NUMBER[number]
    row: dict = {"number": number, "task_id": spec.task_id, "scene": spec.scene, "goals": len(spec.goals)}
    t0 = time.time()
    try:
        with headless():
            env = GoalFrankaEnv(spec, perturbation_mode=suite)
            row["budget"] = env.max_episode_steps
            row.update(_audit_starts(env, number, resets))
            if floor_episodes:
                row["floor_successes"] = _floors(env, spec.task_id, suite, number, floor_episodes)
            outcomes = []
            for j in range(episodes):
                actions, _, _, dones = GoalOracle(env).generate_episode(seed=SEED_BLOCK + 1000 * number + j)
                success = bool(dones and dones[-1])
                outcome = {"success": success, "steps": len(actions), "goals_reached": len(env.first_true),
                           "fallback": bool(env.start_fallback)}
                if not success:
                    outcome["unmet"] = [env.goal_report(i) for i in range(len(spec.goals)) if not env.goal_holds(i)]
                outcomes.append(outcome)
            env.close()
        row["episodes"] = outcomes
        row["oracle_successes"] = sum(o["success"] for o in outcomes)
    except Exception as exc:  # a task that cannot even be built is a validation failure, not a crash of the run
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _numbers(selection: str) -> list[int]:
    from afford_pro.bench.tasks import TASK_MAP
    if selection == "all":
        return list(range(1, 211))
    out = []
    for token in selection.split(","):
        if "-" in token and token.replace("-", "").isdigit():
            lo, hi = token.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token) if token.isdigit() else TASK_MAP[token].number)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="all", help="all, or comma-separated numbers, ranges (1-30) or task ids")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--resets", type=int, default=10)
    parser.add_argument("--suite", default="nominal")
    parser.add_argument("--floor-episodes", type=int, default=0, help="random- and null-policy episodes per task")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    numbers = _numbers(args.tasks)
    rows = []
    with Pool(args.workers) as pool:
        jobs = [(n, args.episodes, args.resets, args.suite, args.floor_episodes) for n in numbers]
        for row in pool.imap_unordered(validate, jobs):
            rows.append(row)
            status = row.get("error") or f"{row['oracle_successes']}/{args.episodes}"
            print(f"#{row['number']:03d} {row['task_id']:42s} {status}  ({row['seconds']}s)", flush=True)
    rows.sort(key=lambda r: r["number"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from afford_pro.provenance import collect_provenance

    args.out.write_text(json.dumps({"suite": args.suite, "episodes": args.episodes, "resets": args.resets,
                                    "floor_episodes": args.floor_episodes, "seed_block": SEED_BLOCK,
                                    "provenance": collect_provenance(), "tasks": rows}, indent=1, default=str),
                        encoding="utf-8")
    ok = sum(r.get("oracle_successes", 0) for r in rows)
    print(f"oracle {ok}/{len(rows) * args.episodes}; build errors {sum('error' in r for r in rows)}; wrote {args.out}")


if __name__ == "__main__":
    main()
