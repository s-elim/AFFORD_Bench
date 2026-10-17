"""Runtime audit of every core task: what the scene actually contains at the goal and at reset.

For each task (headless, no rendering) the audit records:
  - goal calibration: resting vs hold goal, settle drift, tilt, graspable at the goal, and the body under the goal
    (so a reviewer can compare "place it into the red bin" with the body the success predicate refers to);
  - reach tasks: the body under the target point and the gap above it;
  - resets over --resets seeds: start fallback rate, spread of the evaluated body's start, success at reset, and
    the smallest distractor distance to the start and goal;
  - the task tier (afford_pro.task_tiers).

Usage:
  python scripts/audit_tasks.py --out docs/roadmap/evidence/task_audit_2026-10-17.json --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_SEED = 300_000  # disjoint from generation (42 + 1000n + k) and evaluation (200000 + 1000n + j) seeds


def _body_below(env: Any, point: np.ndarray, exclude_body: int) -> tuple[str | None, float | None]:
    """Root body name of the first colliding non-robot geom straight below `point`, and the gap to it (m)."""
    m, d = env.model, env.data
    geom = np.zeros(1, dtype=np.int32)
    start = np.asarray(point, dtype=np.float64) + np.array([0.0, 0.0, 0.002])
    for _ in range(10):
        dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), None, 1, exclude_body, geom)
        if dist < 0:
            return None, None
        g = int(geom[0])
        hit_z = start[2] - dist
        body = int(m.geom_bodyid[g])
        if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body_rootid[body] != env.robot_root_id:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
            return name, float(point[2] - hit_z)
        start = np.array([start[0], start[1], hit_z - 1e-4])
    return None, None


def _distractor_gap(env: Any, start: np.ndarray, goal: np.ndarray) -> float | None:
    gaps = []
    for b in env.free_body_qposadr:
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name and name.startswith("distractor"):
            xy = env.data.xpos[b][:2]
            gaps.append(min(np.linalg.norm(xy - start[:2]), np.linalg.norm(xy - goal[:2])))
    return float(min(gaps)) if gaps else None


def _reset_stats(env: Any, resets: int, seed: int) -> dict[str, Any]:
    starts, fallbacks, at_reset, gaps = [], [], [], []
    for k in range(resets):
        env.reset(seed=seed + k)
        fallbacks.append(bool(getattr(env, "start_fallback", False)))
        at_reset.append(bool(env.check_success()))
        if env.target_qposadr is not None:
            start = env.data.xpos[env.target_id].copy()
            starts.append(start)
            if env.use_distractors:
                gaps.append(_distractor_gap(env, start, env.goal_position))
    out: dict[str, Any] = {"start_fallback_rate": float(np.mean(fallbacks)), "success_at_reset": int(sum(at_reset))}
    if starts:
        xy = np.array(starts)[:, :2]
        out["start_xy_std_m"] = xy.std(axis=0).round(4).tolist()
        out["start_offset_max_m"] = float(np.linalg.norm(xy - env.nominal_obj_pos[:2], axis=1).max())
        out["start_goal_dist_min_m"] = float(np.linalg.norm(xy - env.goal_position[:2], axis=1).min())
    if gaps:
        out["distractor_gap_min_m"] = float(min(g for g in gaps if g is not None))
    return out


def audit_task(task_id: str, resets: int) -> dict[str, Any]:
    import afford_pro
    from afford_pro.envs.headless import headless
    from afford_pro.task_tiers import task_tier

    entry = afford_pro.get_task_entry(task_id)
    with headless():
        env = afford_pro.make(task_id, perturbation_mode="nominal")
        try:
            row: dict[str, Any] = {"task_number": entry.task_number, "task_id": task_id, "domain": entry.domain,
                                   "tier": task_tier(task_id), "task_type": env.task_type,
                                   "object": env.object_name, "instruction": entry.instruction,
                                   "tolerance_xy_m": env.tolerance_xy, "min_lift_m": env.min_lift}
            if env.task_type == "press_button":
                env.reset(seed=AUDIT_SEED)
                body, gap = _body_below(env, env.press_goal, -1)
                row.update(press_goal=np.round(env.press_goal, 4).tolist(), body_below_goal=body, gap_above_body_m=gap)
            elif env.target_qposadr is not None:
                cal = env.calibration
                row.update(hold_goal=bool(env.hold_goal), goal=np.round(env.goal_position, 4).tolist(),
                           goal_drift_m=float(cal["drift"]), goal_tilt_deg=float(cal["tilt_deg"]),
                           graspable_at_goal=bool(cal["graspable"]))
                env.reset(seed=AUDIT_SEED)
                body, gap = _body_below(env, env.goal_position, env.target_id)
                row.update(body_below_goal=body, gap_below_goal_m=gap)
            row.update(_reset_stats(env, resets, AUDIT_SEED))
        finally:
            env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "roadmap" / "evidence" / "task_audit.json")
    parser.add_argument("--resets", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()

    import afford_pro
    from afford_pro.provenance import collect_provenance

    tasks = afford_pro.list_tasks() if args.tasks == "all" else args.tasks.split(",")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(audit_task, tasks, [args.resets] * len(tasks)))
    for r in rows:
        print(f"{r['task_number']:3d} {r['task_id']:36s} {r['tier']:20s} below_goal={r.get('body_below_goal')} "
              f"fallback={r['start_fallback_rate']:.2f} std={r.get('start_xy_std_m')}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"provenance": collect_provenance(), "audit_seed": AUDIT_SEED,
                                    "resets": args.resets, "tasks": rows}, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
