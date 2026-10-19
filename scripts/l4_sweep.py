"""Oracle success as a function of contact friction and object mass (calibration of the L4 physical suite).

For each task and each (mu, mass scale) cell, the L4 perturbation is applied with a degenerate range, i.e. exactly
that sliding friction on the evaluated body and both finger pads and exactly that mass/inertia scale, and the
privileged oracle runs --seeds episodes. A suite that does not lower the expert's success at any setting cannot
show that the physical axis matters; this sweep finds where it starts to.

  python scripts/l4_sweep.py --tier core --mu 0.02,0.05,0.12,0.25 --mass 1.0,2.5,5.0 --seeds 3 --workers 8 \
      --out docs/roadmap/evidence/l4_sweep_2026-10-17.csv
"""
# Deprecated legacy tool: runs the single-body legacy task set (task_set="legacy") to reproduce its evidence.
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWEEP_SEED = 400_000  # disjoint from generation (42 + 1000n + k), evaluation (200000 + ...), and audit (300000 + k)


def run_cell(task_id: str, mu: float | None, mass: float, seeds: int) -> list[dict]:
    import afford_pro
    from afford_pro.envs.headless import headless

    rows = []
    with headless():
        mode = "nominal" if mu is None and mass == 1.0 else "p4_physical"
        env = afford_pro.make(task_id, perturbation_mode=mode, task_set="legacy")
        cfg = env.perturbation_engine.config
        if mode == "p4_physical":
            cfg.friction_range = (mu, mu) if mu is not None else (float("nan"), float("nan"))
            cfg.mass_scale_range = (mass, mass)
        oracle = afford_pro.get_oracle(task_id, env)
        for k in range(seeds):
            actions, _, _, _ = oracle.generate_episode(seed=SWEEP_SEED + k)
            rows.append({"task_id": task_id, "mu": "nominal" if mu is None else mu, "mass_scale": mass,
                         "seed": SWEEP_SEED + k, "success": bool(env.check_success()), "steps": len(actions)})
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="core")
    parser.add_argument("--tasks", default=None, help="comma-separated task ids (overrides --tier)")
    parser.add_argument("--mu", default="0.02,0.05,0.12,0.25")
    parser.add_argument("--mass", default="1.0,2.5,5.0")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from afford_pro.provenance import provenance_json
    from afford_pro.task_tiers import tasks_in_tier

    tasks = args.tasks.split(",") if args.tasks else tasks_in_tier(args.tier)
    cells = [(None, 1.0)] + list(itertools.product([float(x) for x in args.mu.split(",")],
                                                   [float(x) for x in args.mass.split(",")]))
    jobs = [(t, mu, mass) for t in tasks for mu, mass in cells]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cell, t, mu, mass, args.seeds) for t, mu, mass in jobs]
        rows = [r for f in futures for r in f.result()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".provenance.json").write_text(provenance_json(), encoding="utf-8")
    print(f"Wrote {len(rows)} episodes to {args.out}")


if __name__ == "__main__":
    main()
