"""Reactive perturbation probe: does a policy react to an object displaced mid-approach?

Three policies share one privileged feedback law (servo the TCP over the object, then down to its grasp pose,
gripper open) and differ only in how often they replan:

  H = 1          closed loop: one action from the current state at every step;
  H = 16         action chunking (ACT / Diffusion Policy style): every 16 steps, the law is rolled out for 16
                 steps on a copy of the current state and the resulting actions are executed without feedback;
  H = T          open-loop replay: all T approach actions are planned from the initial state.

At t = T/2 the object is displaced by +-4 cm along x or y (pose shifted, velocity zeroed). After T approach
steps the gripper closes and the arm lifts 5 cm. Reported per policy: final TCP-object xy error at closing,
grasp-and-lift success, and the number of steps after the displacement until the TCP is back within
1 cm (xy) of the object (recovery; -1 if it never is).

Usage:
  python scripts/test_reactive_perturbation.py --asset-dir <assets> --tasks tabletop_sort_red,industrial_pin_to_tray
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sanity_baselines import _disable_rendering, _redirect_assets  # noqa: E402

HOVER = 0.10            # pre-grasp height above the grasp pose (m)
MAX_STEP = 0.02         # TCP displacement cap of the feedback law per 50 ms step (m)
DESCEND_XY_TOL = 0.002  # descend once the TCP is this close to the object in xy (m) ...
DESCEND_YAW_TOL = np.radians(3.0)  # ... and the hand yaw is aligned with the grasp yaw
SLOW_STEP = 0.006       # step cap in the last 3 cm above the grasp pose (m)
ABORT_XY_TOL = 0.015    # while descending, return to the hover height only beyond this xy error (m)
RECOVERED_XY = 0.01     # TCP back within this xy distance of the displaced object counts as recovered (m)


@dataclass
class Result:
    task_id: str
    seed: int
    displacement_mm: str
    horizon: int
    xy_error_mm: float
    success: bool
    recovery_steps: int


class ApproachLaw:
    """Stateless feedback law: TCP to HOVER above the grasp pose, then down to it, gripper open at `finger`."""

    def __init__(self, env, dz: float, yaw: float, finger: float):
        self.env, self.dz, self.yaw, self.grip = env, dz, yaw, float(np.clip(2.0 * finger / 0.04 - 1.0, -1, 1))

    def target(self) -> np.ndarray:
        obj = self.env.data.xpos[self.env.target_id]
        tcp = self.env.tcp_position
        # Hysteresis: start descending within DESCEND_XY_TOL; once below the hover height, climb back only if the
        # object is more than ABORT_XY_TOL away (e.g. after a displacement).
        descending = tcp[2] < obj[2] + self.dz + HOVER - 0.01
        yaw_err = abs((self.yaw - self.env.tcp_yaw + np.pi) % (2 * np.pi) - np.pi)
        ready = np.linalg.norm(tcp[:2] - obj[:2]) <= DESCEND_XY_TOL and yaw_err < DESCEND_YAW_TOL
        hover = np.linalg.norm(tcp[:2] - obj[:2]) > ABORT_XY_TOL if descending else not ready
        return obj + np.array([0.0, 0.0, self.dz + (HOVER if hover else 0.0)])

    def action(self) -> np.ndarray:
        env = self.env
        target = self.target()
        delta = target - env.tcp_position
        cap = SLOW_STEP if env.tcp_position[2] - target[2] < 0.03 else MAX_STEP  # slow final descent
        n = np.linalg.norm(delta)
        if n > cap:
            delta *= cap / n
        a = np.zeros(7)
        a[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
        a[5] = np.clip(((self.yaw - env.tcp_yaw + np.pi) % (2 * np.pi) - np.pi) / env.yaw_scale, -1.0, 1.0)
        a[6] = self.grip
        return a


def _sync(dst, src) -> None:
    """Copy the simulation state of env `src` into env `dst` (same task, same model)."""
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        getattr(dst.data, name)[:] = getattr(src.data, name)
    dst.data.time = src.data.time
    dst.finger_cmd = src.finger_cmd
    mujoco.mj_forward(dst.model, dst.data)


def _plan_chunk(scratch, env, law_scratch, steps: int) -> list[np.ndarray]:
    """Roll the law out for `steps` steps on a copy of the current state and return its actions."""
    _sync(scratch, env)
    actions = []
    for _ in range(steps):
        a = law_scratch.action()
        actions.append(a)
        scratch.step(a)
    return actions


def _displace(env, offset: np.ndarray) -> bool:
    """Shift the body by `offset` in xy at rest; False if the new pose penetrates a fixture or leaves its support."""
    adr = env.target_qposadr
    saved = env.data.qpos.copy()
    env.data.qpos[adr:adr + 2] += offset
    dof = env.free_qpos_to_dof[adr]
    env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    hits = env._footprint_hits()
    lo, _ = env.object_extent_z()
    penetrating = any(c.dist < -0.001 for c in env.data.contact[: env.data.ncon]
                      if env.target_id in (env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]))
    valid = bool(hits) and max(hits) - min(hits) <= 0.002 and abs(lo - max(hits)) < 0.003 and not penetrating
    if not valid:
        env.data.qpos[:] = saved
        mujoco.mj_forward(env.model, env.data)
    return valid


def run_episode(env, scratch, oracle, task_id: str, seed: int, offset: np.ndarray, horizon: int, T: int) -> Result | None:
    env.reset(seed=seed)
    dz, yaw, finger = oracle._grasp_plan()
    law_scratch = ApproachLaw(scratch, dz, yaw, finger)
    queue: list[np.ndarray] = []
    recovery = -1
    for t in range(T):
        if t == T // 2:
            if not _displace(env, offset):
                return None
        if not queue:
            queue = _plan_chunk(scratch, env, law_scratch, min(horizon, T - t))
        env.step(queue.pop(0))
        if t >= T // 2 and recovery < 0:
            if np.linalg.norm(env.tcp_position[:2] - env.data.xpos[env.target_id][:2]) < RECOVERED_XY:
                recovery = t + 1 - T // 2
    tcp, obj = env.tcp_position, env.data.xpos[env.target_id]
    error = float(np.linalg.norm(tcp[:2] - obj[:2]))
    z0 = float(obj[2])
    hold = env.tcp_position.copy()
    for _ in range(12):
        a = np.zeros(7)
        a[:3] = np.clip((hold - env.tcp_position) / env.action_scale, -1, 1)
        a[6] = -1.0
        env.step(a)
    for k in range(10):
        a = np.zeros(7)
        a[:3] = np.clip((hold + np.array([0, 0, 0.005 * (k + 1)]) - env.tcp_position) / env.action_scale, -1, 1)
        a[6] = -1.0
        env.step(a)
    lifted = float(env.data.xpos[env.target_id][2]) - z0 > 0.03 and env.is_grasped(env.target_id)
    label = f"{offset[0] * 1000:+.0f},{offset[1] * 1000:+.0f}"
    return Result(task_id, seed, label, horizon, error * 1000.0, bool(lifted), recovery)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=str, default="tabletop_sort_red,tabletop_sort_blue,industrial_pin_to_tray")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--approach-steps", type=int, default=60, help="T; the displacement is applied at T/2")
    parser.add_argument("--chunk", type=int, default=16, help="Action-chunk horizon H of the chunking policy")
    parser.add_argument("--displacement", type=float, default=0.04, help="|dx| or |dy| (m)")
    parser.add_argument("--asset-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV of per-episode results")
    args = parser.parse_args()

    _disable_rendering()
    if args.asset_dir is not None:
        _redirect_assets(args.asset_dir.resolve())
    import afford_pro

    d = args.displacement
    offsets = [np.array(v) for v in ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d))]
    horizons = {"closed loop (H=1)": 1, f"chunked (H={args.chunk})": args.chunk,
                f"open-loop replay (H={args.approach_steps})": args.approach_steps}
    results: list[Result] = []
    for task_id in args.tasks.split(","):
        env = afford_pro.make(task_id, perturbation_mode="nominal")
        scratch = afford_pro.make(task_id, perturbation_mode="nominal")
        oracle = afford_pro.get_oracle(task_id, env)
        for seed in range(args.seeds):
            for offset in offsets:
                for horizon in horizons.values():
                    r = run_episode(env, scratch, oracle, task_id, 30000 + seed, offset, horizon, args.approach_steps)
                    if r is not None:
                        results.append(r)
    print(f"{'policy':28s} {'episodes':>8s} {'success':>8s} {'xy error mm (median, max)':>26s} {'recovery steps (median)':>24s}")
    for name, h in horizons.items():
        rs = [r for r in results if r.horizon == h]
        if not rs:
            continue
        err = np.array([r.xy_error_mm for r in rs])
        rec = [r.recovery_steps for r in rs if r.recovery_steps >= 0]
        latency = f"{np.median(rec):.0f} ({len(rec)}/{len(rs)} recover)" if rec else f"none (0/{len(rs)})"
        print(f"{name:28s} {len(rs):8d} {np.mean([r.success for r in rs]):8.1%} {np.median(err):12.1f}, {err.max():8.1f}   {latency:>24s}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
