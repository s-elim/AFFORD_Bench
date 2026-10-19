"""Sanity baselines for Table II: oracle, random, and null policies on every task.

For each task in the nominal suite this script records:
  - success_at_reset: the success predicate already holds before any action
  - oracle_sr:        success rate of the closed-loop oracle over --oracle-episodes reset seeds
  - oracle_sr_physical: the same under the L4 physical suite (low friction, heavy body)
  - random_sr:        success rate of uniform random actions over --episodes episodes
  - null_sr:          success rate of a policy that never moves and keeps the gripper open
  - reset_pose_varies: the evaluated body pose or the initial TCP position differs across reset seeds
  - start_fallback_rate: fraction of those resets that found no valid scattered start and used the nominal pose

Rendering is disabled, so the script runs on headless machines without OpenGL.

Usage:
  python scripts/sanity_baselines.py --out docs/reviews/sanity_baselines.csv
  # Use a different asset directory (e.g. XMLs with a portable meshdir):
  python scripts/sanity_baselines.py --asset-dir /path/to/assets --out sanity.csv
  # Run one of n shards (merge the CSVs afterwards):
  python scripts/sanity_baselines.py --shard 0/4 --out part0.csv
"""
# Deprecated legacy tool: runs the single-body legacy task set (task_set="legacy") to reproduce its evidence.
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

import afford_pro.envs.base_franka_env as base_env

NULL_ACTION = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)


@dataclass
class TaskSanity:
    task_number: int
    task_id: str
    domain: str
    success_at_reset: bool
    oracle_sr: float
    oracle_sr_physical: float
    random_sr: float
    null_sr: float
    reset_pose_varies: bool
    start_fallback_rate: float
    # Oracle success within the evaluation budget H (the oracle itself may run up to BaseOracle.max_steps).
    oracle_sr_h: float = float("nan")
    oracle_sr_physical_h: float = float("nan")
    oracle_mean_steps: float = float("nan")
    oracle_max_steps: int = -1
    random_episodes: int = -1


@dataclass
class OracleEpisode:
    task_id: str
    mode: str
    seed: int
    success: bool
    steps: int
    start_fallback: bool


class _NoRenderer:
    def __init__(self, *args, **kwargs) -> None:
        pass


def _disable_rendering() -> None:
    base_env.mujoco.Renderer = _NoRenderer
    base_env.BaseFrankaEnv.get_observation = lambda self: {"tcp_pose": self.tcp_pose}


def _redirect_assets(asset_dir: Path) -> None:
    import afford_pro.envs.parametric_env as parametric_env

    parametric_env.ASSET_DIR = asset_dir


def _rollout_sr(env, policy: str, episodes: int, horizon: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    successes = 0
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        for _ in range(horizon):
            action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32) if policy == "random" else NULL_ACTION
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                successes += 1
                break
    return successes / episodes


def _oracle_episodes(task_id: str, mode: str, episodes: int, seed: int) -> list[OracleEpisode]:
    import afford_pro

    env = afford_pro.make(task_id, perturbation_mode=mode, task_set="legacy")
    oracle = afford_pro.get_oracle(task_id, env)
    records = []
    for ep in range(episodes):
        actions, _, _, _ = oracle.generate_episode(seed=seed + ep)
        records.append(OracleEpisode(task_id, mode, seed + ep, bool(env.check_success()), len(actions),
                                     bool(getattr(env, "start_fallback", False))))
    return records


def _sr(records: list[OracleEpisode], horizon: int | None = None) -> float:
    return float(np.mean([r.success and (horizon is None or r.steps <= horizon) for r in records]))


def evaluate_task(task_id: str, episodes: int, oracle_episodes: int, horizon: int, seed: int,
                  episode_log: list[OracleEpisode] | None = None) -> TaskSanity:
    import afford_pro

    entry = afford_pro.get_task_entry(task_id, "legacy")
    env = afford_pro.make(task_id, perturbation_mode="nominal", task_set="legacy")
    body = env.target_id

    poses = []
    reset_success = []
    fallbacks = []
    for s in (seed, seed + 1, seed + 2):
        env.reset(seed=s)
        poses.append(np.concatenate([env.data.xpos[body], env.tcp_position]))
        reset_success.append(env.check_success())
        fallbacks.append(env.start_fallback)

    nominal = _oracle_episodes(task_id, "nominal", oracle_episodes, seed)
    physical = _oracle_episodes(task_id, "p4_physical", oracle_episodes, seed)
    if episode_log is not None:
        episode_log.extend(nominal + physical)
    return TaskSanity(
        task_number=entry.task_number,
        task_id=task_id,
        domain=entry.domain,
        success_at_reset=any(reset_success),
        oracle_sr=_sr(nominal),
        oracle_sr_physical=_sr(physical),
        random_sr=_rollout_sr(env, "random", episodes, horizon, seed),
        null_sr=_rollout_sr(env, "null", 1, horizon, seed),
        reset_pose_varies=not all(np.allclose(poses[0], p) for p in poses[1:]),
        start_fallback_rate=float(np.mean(fallbacks)),
        oracle_sr_h=_sr(nominal, horizon),
        oracle_sr_physical_h=_sr(physical, horizon),
        oracle_mean_steps=float(np.mean([r.steps for r in nominal])),
        oracle_max_steps=int(max(r.steps for r in nominal)),
        random_episodes=episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "sanity_baselines.csv")
    parser.add_argument("--asset-dir", type=Path, default=None, help="Override afford_pro/assets")
    parser.add_argument("--episodes", type=int, default=5, help="Random-policy episodes per task")
    parser.add_argument("--oracle-episodes", type=int, default=5, help="Oracle episodes per task and suite")
    parser.add_argument("--shard", type=str, default="0/1", help="i/n: evaluate every n-th task starting at i")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated task IDs, or 'all'")
    parser.add_argument("--episodes-out", type=Path, default=None,
                        help="Also write one CSV row per oracle episode (task, suite, seed, success, steps)")
    args = parser.parse_args()

    _disable_rendering()
    if args.asset_dir is not None:
        _redirect_assets(args.asset_dir.resolve())
    import afford_pro
    from afford_pro.provenance import provenance_json

    task_ids = afford_pro.list_tasks("legacy") if args.tasks == "all" else args.tasks.split(",")
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    task_ids = task_ids[shard::n_shards]
    rows = []
    episode_log: list[OracleEpisode] = []
    for tid in task_ids:
        row = evaluate_task(tid, args.episodes, args.oracle_episodes, args.horizon, args.seed, episode_log)
        rows.append(row)
        print(f"[{row.task_number:03d}] {tid}: oracle={row.oracle_sr:.2f} (H={args.horizon}: {row.oracle_sr_h:.2f}) "
              f"oracle_L4={row.oracle_sr_physical:.2f} random={row.random_sr:.2f} "
              f"null={row.null_sr:.2f} reset_success={row.success_at_reset} pose_varies={row.reset_pose_varies}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    if args.episodes_out is not None:
        args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
        with args.episodes_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(episode_log[0])))
            writer.writeheader()
            writer.writerows(asdict(r) for r in episode_log)
    args.out.with_suffix(".provenance.json").write_text(provenance_json(extra={"shard": args.shard}), encoding="utf-8")

    by_domain: dict[str, list[TaskSanity]] = {}
    for r in rows:
        by_domain.setdefault(r.domain, []).append(r)
    print("\nDomain                                   Oracle  Or.L4  Random   Null")
    for domain, rs in by_domain.items():
        print(f"{domain:40s} {np.mean([r.oracle_sr for r in rs]):6.1%} {np.mean([r.oracle_sr_physical for r in rs]):6.1%} "
              f"{np.mean([r.random_sr for r in rs]):6.1%} {np.mean([r.null_sr for r in rs]):6.1%}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
