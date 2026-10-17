"""Aggregate sharded `sanity_baselines.py` output into Table II candidates with uncertainty.

Reads part_*.csv (per task) and episodes_*.csv (per oracle episode) from --dir and writes summary.json and summary.md:
success rates per domain and per task tier for the oracle (nominal and L4, within the oracle budget and within H),
random, and null policies, each with an episode-level Wilson interval and a task-clustered bootstrap interval.

  python scripts/aggregate_sanity.py --dir docs/roadmap/evidence/sanity_2026-10-17
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afford_pro.evaluation.stats import summarize
from afford_pro.task_tiers import task_tier

TRUE = {"True", "true", "1"}


def _read(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for p in paths:
        with p.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _oracle_clusters(episodes: list[dict[str, str]], mode: str, horizon: int | None, keep) -> list[list[bool]]:
    per_task: dict[str, list[bool]] = defaultdict(list)
    for e in episodes:
        if e["mode"] == mode and keep(e["task_id"]):
            ok = e["success"] in TRUE and (horizon is None or int(e["steps"]) <= horizon)
            per_task[e["task_id"]].append(ok)
    return list(per_task.values())


def _rate_clusters(tasks: list[dict[str, str]], column: str, n_eps: int, keep) -> list[list[bool]]:
    """Per-task rates from the task CSV expanded to episodes (k successes of n)."""
    out = []
    for t in tasks:
        if keep(t["task_id"]):
            n = n_eps if column == "null_sr" else int(t.get("random_episodes") or -1)
            n = n_eps if n < 0 else n
            k = round(float(t[column]) * n)
            out.append([True] * k + [False] * (n - k))
    return out


def _group_summary(tasks, episodes, keep, horizon, n_random) -> dict[str, dict]:
    return {
        "oracle_nominal_budget": summarize(_oracle_clusters(episodes, "nominal", None, keep)),
        f"oracle_nominal_H{horizon}": summarize(_oracle_clusters(episodes, "nominal", horizon, keep)),
        "oracle_L4_budget": summarize(_oracle_clusters(episodes, "p4_physical", None, keep)),
        f"oracle_L4_H{horizon}": summarize(_oracle_clusters(episodes, "p4_physical", horizon, keep)),
        "random": summarize(_rate_clusters(tasks, "random_sr", n_random, keep)),
        "null": summarize(_rate_clusters(tasks, "null_sr", 1, keep)),
    }


def _fmt(s: dict) -> str:
    lo, hi = s["task_cluster_bootstrap95"]
    return f"{100 * s['rate']:.1f} ({100 * lo:.1f}-{100 * hi:.1f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--random-episodes", type=int, default=5)
    args = parser.parse_args()

    tasks = _read(sorted(args.dir.glob("part_*.csv")))
    episodes = _read(sorted(args.dir.glob("episodes_*.csv")))
    domain_of = {t["task_id"]: t["domain"] for t in tasks}
    groups = {f"all {len(tasks)}": lambda tid: True}
    groups.update({f"domain: {d}": (lambda tid, d=d: domain_of[tid] == d) for d in dict.fromkeys(domain_of.values())})
    groups.update({f"tier: {tr}": (lambda tid, tr=tr: task_tier(tid) == tr) for tr in ("core", "reach", "instruction_mismatch")})
    summary = {name: _group_summary(tasks, episodes, keep, args.horizon, args.random_episodes) for name, keep in groups.items()}
    failures = sorted({(e["task_id"], e["mode"]) for e in episodes if e["success"] not in TRUE})
    over_h = sorted({e["task_id"] for e in episodes
                     if e["mode"] == "nominal" and e["success"] in TRUE and int(e["steps"]) > args.horizon})
    out = {"n_tasks": len(tasks), "n_oracle_episodes": len(episodes), "groups": summary,
           "success_at_reset_tasks": [t["task_id"] for t in tasks if t["success_at_reset"] in TRUE],
           "oracle_nominal_failure_tasks": [t for t, m in failures if m == "nominal"],
           "oracle_nominal_over_horizon_tasks": over_h}
    (args.dir / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    h = args.horizon
    if len(tasks) != len({t['task_id'] for t in tasks}):
        raise SystemExit('duplicate task rows across shards')
    lines = ["| Group | Oracle | Oracle (H=%d) | Oracle L4 | Oracle L4 (H=%d) | Random | Null |" % (h, h),
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in summary.items():
        lines.append(f"| {name} | {_fmt(s['oracle_nominal_budget'])} | {_fmt(s[f'oracle_nominal_H{h}'])} | "
                     f"{_fmt(s['oracle_L4_budget'])} | {_fmt(s[f'oracle_L4_H{h}'])} | {_fmt(s['random'])} | {_fmt(s['null'])} |")
    note = ("Rates in %; parentheses: 95% task-clustered bootstrap interval. Oracle: 5 episodes per task and suite, "
            f"budget = oracle's own 600-step limit. Random: 5 episodes, null: 1 episode, H = {h}.\n")
    (args.dir / "summary.md").write_text(note + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print({k: v for k, v in out.items() if k != "groups"})


if __name__ == "__main__":
    main()
