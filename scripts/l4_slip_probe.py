"""L4 slip probe: does a grasp away from the center of mass slip under the physical (L4) suite?

For each task, suite (nominal, p4_physical), lateral grasp offset, and seed, the probe plans the oracle's
grasp, shifts it by the offset along the hand's lateral axis (perpendicular to the closing axis), closes,
lifts 10 cm, carries 20 cm sideways at the speed cap, and records whether the body is still held and how far
it moved (translation, mm) and turned (rotation, deg) relative to the hand.

Usage:
  python scripts/l4_slip_probe.py --tasks tabletop_sort_red,kitchen_mug_return_bin --out slip.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sanity_baselines import _disable_rendering, _redirect_assets  # noqa: E402


def _probe(env, offset: float, seed: int) -> dict:
    import afford_pro.oracles.parametric_oracle as po
    from afford_pro.oracles.base_oracle import EpisodeDone

    result: dict = {}

    class OffsetGrasp(po.ParametricOracle):
        def run(self) -> None:
            dz, yaw, finger = self._grasp_plan()
            off = offset * np.array([np.cos(yaw), np.sin(yaw), 0.0])
            grip = po._grip_command(finger)
            self.servo(lambda: self._obj() + off + np.array([0, 0, dz + 0.10]), yaw, grip, tol=0.003, max_steps=120)
            self.servo(lambda: self._obj() + off + np.array([0, 0, dz]), yaw, grip, tol=0.0015, max_steps=80, max_step=0.012)
            grasp_tcp = self.env.tcp_position.copy()
            for _ in range(12):
                self.act(self.action_towards(grasp_tcp, yaw, po.CLOSE))
            result["grasped"] = bool(self.env.is_grasped(self.env.target_id))
            if not result["grasped"]:
                raise EpisodeDone
            hand_rot = lambda: self.env.data.xmat[self.env.hand_id].reshape(3, 3)  # noqa: E731
            rel = lambda: hand_rot().T @ (self._obj() - self.env.tcp_position)  # noqa: E731
            rot0 = hand_rot().T @ self.env.data.xmat[self.env.target_id].reshape(3, 3)
            rel0 = rel()
            start = self.env.tcp_position.copy()
            self.servo(start + np.array([0, 0, 0.10]), yaw, po.CLOSE, tol=0.003, max_steps=40)
            side = -0.20 if start[1] > 0 else 0.20
            self.servo(start + np.array([0.0, side, 0.10]), yaw, po.CLOSE, tol=0.003, max_steps=60)
            self.hold(po.CLOSE, 10, yaw)
            rot1 = hand_rot().T @ self.env.data.xmat[self.env.target_id].reshape(3, 3)
            cos = np.clip((np.trace(rot0.T @ rot1) - 1.0) / 2.0, -1.0, 1.0)
            result.update(held=bool(self.env.is_grasped(self.env.target_id)),
                          slip_mm=float(np.linalg.norm(rel() - rel0) * 1000.0), rot_deg=float(np.degrees(np.arccos(cos))))
            raise EpisodeDone

    OffsetGrasp(env).generate_episode(seed=seed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task IDs")
    parser.add_argument("--offsets-mm", type=str, default="0,10,20")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--asset-dir", type=Path, default=None, help="Override afford_pro/assets")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _disable_rendering()
    if args.asset_dir is not None:
        _redirect_assets(args.asset_dir.resolve())
    import afford_pro

    rows = []
    for task_id in args.tasks.split(","):
        for mode in ("nominal", "p4_physical"):
            env = afford_pro.make(task_id, perturbation_mode=mode)
            for offset_mm in (float(x) for x in args.offsets_mm.split(",")):
                for seed in range(args.seeds):
                    r = dict(task_id=task_id, mode=mode, offset_mm=offset_mm, seed=seed, grasped=False, held=False,
                             slip_mm=float("nan"), rot_deg=float("nan"))
                    r.update(_probe(env, offset_mm / 1000.0, seed))
                    rows.append(r)
                    print(r, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
