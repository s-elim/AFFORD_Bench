"""Generate oracle demonstrations for one task (Robomimic HDF5).

Uses the same generator, seed scheme (episode seeds start at --seed + 1000 * task_number), provenance record, and
overwrite protection as scripts/generate_all_150_tasks.py, so a file made here is interchangeable with one made by
the batch script. An existing file at the output path is never replaced unless --overwrite-invalid (invalid files)
or --overwrite (any file) is given.

  python scripts/generate_demos.py --task industrial_shaft_assembly --num-demos 5 --output-dir datasets/scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.datasets.generation import GenerationConfig, generate_task_dataset
from afford_pro.datasets.validation import ENV_HORIZON


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default="industrial_shaft_assembly", choices=afford_pro.list_all_tasks(True))
    parser.add_argument("--num-demos", type=int, default=5, help="Number of successful demonstrations to collect")
    parser.add_argument("--perturbation", type=str, default="nominal",
                        choices=["nominal", "p1_object", "p2_spatial", "p3_visual", "p4_physical", "combined_ood"])
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=42, help="Episode seeds start at seed + 1000 * task_number")
    parser.add_argument("--depth", action="store_true", help="Also record metric depth")
    parser.add_argument("--max-demo-steps", type=int, default=ENV_HORIZON)
    parser.add_argument("--overwrite-invalid", action="store_true", help="Replace an existing file that fails validation")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing file even if it is valid")
    args = parser.parse_args()

    cfg = GenerationConfig(task_id=args.task, output_dir=ROOT / args.output_dir, num_demos=args.num_demos,
                           perturbation=args.perturbation, seed=args.seed, depth=args.depth,
                           max_demo_steps=args.max_demo_steps, overwrite_valid=args.overwrite,
                           overwrite_invalid=args.overwrite or args.overwrite_invalid)
    result = generate_task_dataset(cfg, log=lambda msg: print(msg, flush=True))
    print(f"{result.status}: {result.path} ({result.collected} demos, {result.attempts} attempts, "
          f"{len(result.failed_seeds)} oracle failures, {len(result.over_length_seeds)} over-length; "
          f"validation {result.validation_status})")
    return 0 if result.status in ("written", "skipped_valid") and result.validation_status != "INVALID" else 1


if __name__ == "__main__":
    sys.exit(main())
