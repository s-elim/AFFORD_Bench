"""Evaluate a policy on AFFORD-Bench suites with the protocol's seeds, horizon, and uncertainty reporting.

Episode seeds are `--seed-base + 1000 * task_number + j` (j < --episodes). Within each task they are disjoint from
the generation seeds (`42 + 1000 * task_number + k`, k < 400); numerically equal seeds of different tasks give
unrelated episodes. The same seeds are used in every suite, and model perturbations draw from their own random
stream, so a seed gives the same layout in nominal, p1_object, p3_visual, and p4_physical (and in p2_spatial and
combined_ood): suites are paired. Outputs in --out-dir: episodes.csv (one row per episode), summary.json (per task and per
suite: success with Wilson and task-clustered bootstrap 95% intervals, PRI and GDI against nominal, force
violation rate), provenance including the sha256 of the checkpoint.

  python scripts/evaluate_policy_suite.py --checkpoint checkpoints/diffusion_lowdim_tabletop_core_best.pt \
      --domain tabletop --tier core --suites nominal,p2_spatial,p3_visual,p4_physical,combined_ood --episodes 50
  python scripts/evaluate_policy_suite.py --policy oracle --task tabletop_sort_red --episodes 10
  python scripts/evaluate_policy_suite.py --checkpoint model.pt --domain kitchen --episodes 50

The horizon defaults to each task's step budget
(afford_pro.bench.spec.step_budget: 400 steps for one goal plus 250 per further goal); legacy tasks keep H = 400.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITES = ("nominal", "p1_object", "p2_spatial", "p3_visual", "p4_physical", "combined_ood")
LEGACY_HORIZON = 400
DOMAIN_MATCH = {"industrial": "Industrial", "kitchen": "Culinary", "tabletop": "Tabletop", "lab": "Laboratory",
                "long_horizon": "Long-Horizon", "chain_precision": "Precision Chaining", "chain_service": "Service & Culinary"}


def select_tasks(task: str | None, domain: str, tier: str | None, task_set: str = "bench") -> list[str]:
    import afford_pro
    from afford_pro.task_tiers import tasks_in_tier

    if task:
        return task.split(",")
    tasks = [t for t in afford_pro.list_tasks(task_set)
             if domain == "all" or DOMAIN_MATCH[domain] in afford_pro.get_task_entry(t, task_set).domain]
    return [t for t in tasks if t in set(tasks_in_tier(tier))] if tier else tasks


def _make_policy(args: argparse.Namespace) -> Any:
    from afford_pro.evaluation.rollout import CheckpointPolicy, NullPolicy, RandomPolicy

    if args.checkpoint:
        return CheckpointPolicy(args.checkpoint, device=args.device, replan=args.replan)
    return {"random": RandomPolicy, "null": NullPolicy}[args.policy]() if args.policy != "oracle" else None


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    from contextlib import nullcontext

    from afford_pro.envs.headless import headless

    policy = _make_policy(args)
    needs_images = getattr(policy, "modality", None) == "visual"
    with nullcontext() if needs_images else headless():
        return _evaluate(args, policy)


def _evaluate(args: argparse.Namespace, policy: Any) -> list[dict[str, Any]]:
    import afford_pro
    from afford_pro.evaluation.rollout import run_oracle_episode, run_policy_episode

    records = []
    for task_id in select_tasks(args.task, args.domain, args.tier, args.task_set):
        entry = afford_pro.get_task_entry(task_id, args.task_set)
        fixed = afford_pro.get_initial_states(task_id) if args.use_init_states else None
        for suite in args.suites.split(","):
            env = afford_pro.make(task_id, task_set=args.task_set, perturbation_mode=suite)
            horizon = args.horizon or (env.max_episode_steps if args.task_set == "bench" else LEGACY_HORIZON)
            try:
                oracle = afford_pro.get_oracle(task_id, env) if policy is None else None
                for j in range(args.episodes):
                    seed = args.seed_base + 1000 * entry.task_number + j
                    if oracle is not None:
                        rec = run_oracle_episode(env, oracle, task_id, suite, seed, horizon)
                    else:
                        init = fixed[j % len(fixed)] if fixed is not None and suite == "nominal" else None
                        rec = run_policy_episode(env, policy, task_id, suite, seed, horizon, init)
                    records.append(asdict(rec))
            finally:
                env.close()
            sr = sum(r["success"] for r in records if r["task_id"] == task_id and r["suite"] == suite) / args.episodes
            print(f"#{entry.task_number:03d} {task_id:36s} {suite:13s} SR {100 * sr:5.1f}%", flush=True)
    return records


def summarize_records(records: list[dict[str, Any]], task_set: str = "bench") -> dict[str, Any]:
    from afford_pro.evaluation.stats import summarize

    by_suite: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    fvr: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_suite[r["suite"]][r["task_id"]].append(bool(r["success"]))
        fvr[r["suite"]].append(bool(r["force_violation"]))
    out: dict[str, Any] = {}
    nominal = summarize(list(by_suite["nominal"].values()))["rate"] if "nominal" in by_suite else None
    for suite, per_task in by_suite.items():
        s = summarize(list(per_task.values()))
        s["per_task_rate"] = {t: sum(v) / len(v) for t, v in per_task.items()}
        s["force_violation_rate"] = sum(fvr[suite]) / len(fvr[suite])
        if nominal:  # PRI and GDI are defined only when the nominal success rate is positive
            s["PRI_pct"] = 100.0 * s["rate"] / nominal
            s["GDI_pct"] = 100.0 - s["PRI_pct"]
        if task_set == "legacy":  # the audited legacy subsets name legacy task definitions
            s["subsets"] = _subset_rates(per_task)
        out[suite] = s
    return out


def _subset_rates(per_task: dict[str, list[bool]]) -> dict[str, dict[str, float | int]]:
    """Pooled success on the audited core subsets that are reported separately (afford_pro.task_tiers.CORE_SUBSETS)
    and on the remaining tasks."""
    from afford_pro.task_tiers import CORE_SUBSETS

    out = {}
    flagged: set[str] = set()
    for name, members in CORE_SUBSETS.items():
        hits = [v for t, v in per_task.items() if t in members]
        flagged |= set(members)
        if hits:
            flat = [x for v in hits for x in v]
            out[name] = {"tasks": len(hits), "episodes": len(flat), "rate": sum(flat) / len(flat)}
    rest = [x for t, v in per_task.items() if t not in flagged for x in v]
    if rest:
        out["other"] = {"tasks": sum(t not in flagged for t in per_task), "episodes": len(rest), "rate": sum(rest) / len(rest)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--policy", type=str, default="oracle", choices=["oracle", "random", "null"],
                        help="reference policy when no --checkpoint is given")
    parser.add_argument("--task", type=str, default=None, help="task id(s), comma-separated")
    parser.add_argument("--domain", type=str, default="all", choices=["all", *DOMAIN_MATCH])
    parser.add_argument("--tier", type=str, default=None, choices=[None, "core", "reach", "instruction_mismatch"])
    parser.add_argument("--suites", type=str, default="nominal")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task-set", choices=("bench", "legacy"), default="bench",
                        help="task set (legacy: the deprecated single-body tasks)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="success must come within this many steps (default: the task's step budget; legacy tasks 400)")
    parser.add_argument("--seed-base", type=int, default=200_000)
    parser.add_argument("--replan", type=int, default=None, help="steps executed per predicted chunk (default: chunk)")
    parser.add_argument("--use-init-states", action="store_true", help="nominal suite starts from data/init_states")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    bad = set(args.suites.split(",")) - set(SUITES)
    if bad:
        parser.error(f"unknown suites {sorted(bad)}")
    if args.task_set == "bench" and (args.tier or args.use_init_states):
        parser.error("--tier and --use-init-states apply to the legacy tasks only")
    if args.seed_base < 100_000 or not 0 < args.episodes <= 1000:
        parser.error("--seed-base must be >= 100000 (above every generation seed of the same task) and "
                     "--episodes in 1..1000 (one task's seed block)")

    from afford_pro.provenance import collect_provenance, sha256_file

    records = evaluate(args)
    summary = summarize_records(records, args.task_set)
    for suite, s in summary.items():
        lo, hi = s["task_cluster_bootstrap95"]
        print(f"{suite:13s} SR {100 * s['rate']:5.1f}% (95% task-clustered {100 * lo:.1f}-{100 * hi:.1f}; "
              f"Wilson {100 * s['wilson95'][0]:.1f}-{100 * s['wilson95'][1]:.1f}; {s['episodes']} episodes)", flush=True)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        ckpt_hash = sha256_file(Path(args.checkpoint)) if args.checkpoint else None
        meta = {"settings": {k: str(v) for k, v in vars(args).items()}, "checkpoint_sha256": ckpt_hash,
                "provenance": collect_provenance(), "summary": summary}
        (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
