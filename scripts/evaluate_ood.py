"""Out-of-distribution evaluation across perturbation suites; a wrapper around scripts/evaluate_policy_suite.py.

Suites (afford_pro/perturbations/engine.py): p1_object (color of the evaluated body, +-25%), p2_spatial (start
offset radius 7-12 cm, |yaw| 45-90 deg), p3_visual (light position +-0.5 m, intensity x[0.5, 1.5] clipped at 1),
p4_physical (sliding friction 0.12-0.25 on body and finger pads, mass and inertia x1.8-2.5), combined_ood (all).
`--perturbation all` also runs the nominal suite so PRI and GDI can be computed on paired seeds.

  python scripts/evaluate_ood.py --checkpoint checkpoints/bc_lowdim_tabletop_core_best.pt --domain tabletop --perturbation all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_policy_suite import DOMAIN_MATCH, SUITES, evaluate, summarize_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--policy", type=str, default="checkpoint", choices=["checkpoint", "oracle", "random", "null"])
    parser.add_argument("--domain", type=str, default="industrial", choices=["all", *DOMAIN_MATCH])
    parser.add_argument("--tier", type=str, default=None, choices=[None, "core", "reach", "instruction_mismatch"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--perturbation", type=str, default="all", choices=["all", *SUITES[1:]])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=200_000)
    args = parser.parse_args()
    if args.policy == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required unless --policy is oracle, random, or null")

    suites = ",".join(SUITES) if args.perturbation == "all" else f"nominal,{args.perturbation}"
    suite_args = argparse.Namespace(
        checkpoint=args.checkpoint, policy="oracle" if args.policy == "checkpoint" else args.policy,
        task=args.task, domain=args.domain, tier=args.tier, suites=suites, episodes=args.num_episodes,
        horizon=args.horizon, seed_base=args.seed, replan=None, use_init_states=False, device=args.device)
    for suite, s in summarize_records(evaluate(suite_args)).items():
        pri = f" PRI {s['PRI_pct']:.1f}% GDI {s['GDI_pct']:.1f}%" if "PRI_pct" in s else ""
        print(f"{suite:13s} SR {100 * s['rate']:5.1f}% ({s['episodes']} episodes){pri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
