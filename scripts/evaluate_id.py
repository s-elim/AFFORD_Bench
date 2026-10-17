"""In-distribution (nominal suite) evaluation; a wrapper around scripts/evaluate_policy_suite.py.

Same seeds, horizon, success definition, and statistics as the suite evaluator (see its docstring).

  python scripts/evaluate_id.py --checkpoint checkpoints/bc_lowdim_tabletop_core_best.pt --domain tabletop --num-episodes 50
  python scripts/evaluate_id.py --policy oracle --task industrial_shaft_assembly --num-episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_policy_suite import DOMAIN_MATCH, evaluate, summarize_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--policy", type=str, default="checkpoint", choices=["checkpoint", "oracle", "random", "null"])
    parser.add_argument("--domain", type=str, default="industrial", choices=["all", *DOMAIN_MATCH])
    parser.add_argument("--tier", type=str, default=None, choices=[None, "core", "reach", "instruction_mismatch"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=200_000, help="evaluation seed block base")
    parser.add_argument("--use-init-states", action="store_true")
    args = parser.parse_args()
    if args.policy == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required unless --policy is oracle, random, or null")

    suite_args = argparse.Namespace(
        checkpoint=args.checkpoint, policy="oracle" if args.policy == "checkpoint" else args.policy,
        task=args.task, domain=args.domain, tier=args.tier, suites="nominal", episodes=args.num_episodes,
        horizon=args.horizon, seed_base=args.seed, replan=None, use_init_states=args.use_init_states,
        device=args.device)
    s = summarize_records(evaluate(suite_args))["nominal"]
    lo, hi = s["task_cluster_bootstrap95"]
    print(f"Nominal SR {100 * s['rate']:.1f}% over {s['episodes']} episodes in {s['tasks']} tasks "
          f"(95% task-clustered {100 * lo:.1f}-{100 * hi:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
