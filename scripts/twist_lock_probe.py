"""Family C (Tasks 227-233) oracle probe: success rate and the mechanism quantities behind Phi_C.

Per episode: success, steps, final mechanism angle and axial slide position, and for the cap the peak axial spring
force F_z and the ring travel accumulated below / at or above 20 N; for the key the insertion depth, latch
retraction and bolt opening. `--press-depth` overrides the oracle's axial compression (e.g. 0.005 m, below the
cap threshold) to show that an under-loaded turn leaves the mechanism locked.

Usage:
  python scripts/twist_lock_probe.py --seeds 21 --out docs/reviews/twist_lock_probe_2026-10-08.csv [--asset-dir <assets>]
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sanity_baselines import _disable_rendering, _redirect_assets  # noqa: E402


@dataclass
class Row:
    task_id: str
    seed: int
    press_depth_mm: float
    success: bool
    steps: int
    angle_deg: float
    press_mm: float
    peak_axial_force_n: float
    ring_travel_below_20n_deg: float
    ring_travel_at_20n_deg: float
    insertion_mm: float
    latch_mm: float
    bolt_mm: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=21)
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--press-depth", type=float, default=None)
    parser.add_argument("--asset-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    _disable_rendering()
    if args.asset_dir is not None:
        _redirect_assets(args.asset_dir.resolve())
    import afford_pro
    from afford_pro.oracles.twist_lock_oracle import PRESS_DEPTH, TwistLockOracle
    from afford_pro.task_definitions import DOMAIN_8_FRONTIER_TASKS

    tasks = [t.task_id for t in DOMAIN_8_FRONTIER_TASKS if t.task_type == "twist_lock"]
    if args.tasks != "all":
        tasks = args.tasks.split(",")
    depth = PRESS_DEPTH if args.press_depth is None else args.press_depth
    rows: list[Row] = []
    for task_id in tasks:
        env = afford_pro.make(task_id)
        oracle = TwistLockOracle(env, press_depth=depth)
        for seed in range(args.seeds):
            actions, *_ = oracle.generate_episode(seed=seed)
            key = env.mech.kind == "key"
            rows.append(Row(
                task_id, seed, depth * 1000.0, bool(env.check_success()), len(actions),
                float(np.degrees(env.mechanism_angle)), env.press_position * 1000.0, env.peak_axial_force,
                float(np.degrees(env.low_force_travel)), float(np.degrees(env.forced_travel)),
                env.insertion_depth * 1000.0 if key else 0.0,
                float(env.data.qpos[env.latch_qadr]) * 1000.0 if key else 0.0,
                float(env.data.qpos[env.bolt_qadr]) * 1000.0 if key else 0.0,
            ))
        rs = [r for r in rows if r.task_id == task_id]
        print(f"{task_id:34s} {sum(r.success for r in rs):3d}/{len(rs)}  median steps {np.median([r.steps for r in rs]):.0f}",
              flush=True)
    print(f"all: {sum(r.success for r in rows)}/{len(rows)}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows({k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(r).items()} for r in rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
