"""Peak robot-scene contact force of the privileged oracle, per task (calibration of the force-violation threshold).

Runs one or more oracle episodes per task headlessly and records the largest `info["robot_scene_force"]` (net force
the scene, excluding the manipulated body, exerts on the robot) over each episode.

  python scripts/force_probe.py --episodes 1 --workers 8 --out docs/roadmap/evidence/force_probe_2026-10-17.csv
"""
# Deprecated legacy tool: runs the single-body legacy task set (task_set="legacy") to reproduce its evidence.
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBE_SEED = 500_000  # disjoint from generation, evaluation, audit, and sweep seed blocks


def probe_task(task_id: str, episodes: int) -> list[dict]:
    import afford_pro
    from afford_pro.envs.headless import headless

    rows = []
    with headless():
        env = afford_pro.make(task_id, perturbation_mode="nominal", task_set="legacy")
        oracle = afford_pro.get_oracle(task_id, env)
        step = env.step
        peak = [0.0]

        def recording_step(action):
            result = step(action)
            peak[0] = max(peak[0], float(result[4].get("robot_scene_force", 0.0)))
            return result

        env.step = recording_step
        for k in range(episodes):
            peak[0] = 0.0
            actions, _, _, _ = oracle.generate_episode(seed=PROBE_SEED + k)
            rows.append({"task_id": task_id, "seed": PROBE_SEED + k, "success": bool(env.check_success()),
                         "steps": len(actions), "peak_force_n": peak[0]})
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import afford_pro
    from afford_pro.provenance import provenance_json

    tasks = afford_pro.list_tasks("legacy")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for rs in pool.map(probe_task, tasks, [args.episodes] * len(tasks)) for r in rs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".provenance.json").write_text(provenance_json(), encoding="utf-8")
    peaks = np.array([r["peak_force_n"] for r in rows])
    print(f"{len(rows)} episodes: median {np.median(peaks):.1f} N, p95 {np.percentile(peaks, 95):.1f} N, "
          f"max {peaks.max():.1f} N, > 150 N in {(peaks > 150).sum()}")


if __name__ == "__main__":
    main()
