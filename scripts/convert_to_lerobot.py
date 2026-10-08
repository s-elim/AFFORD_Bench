"""Convert AFFORD-Bench Robomimic HDF5 demonstrations to LeRobot v2.0 datasets (one dataset per HDF5 file).

Images are stored losslessly as PNG inside the parquet files (LeRobot feature dtype "image"), so the
conversion round-trips exactly (see tests/test_converters.py).

Usage:
  python scripts/convert_to_lerobot.py --hdf5 datasets/industrial_shaft_assembly_nominal.hdf5 --out lerobot/
  python scripts/convert_to_lerobot.py --hdf5 datasets/*.hdf5 --out lerobot/ --robot-type franka_panda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afford_pro.datasets.converters import FPS, read_robomimic, write_lerobot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, nargs="+", required=True, help="Robomimic HDF5 files")
    parser.add_argument("--out", type=Path, required=True, help="Output directory; one dataset per input file")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--robot-type", type=str, default="franka_panda")
    args = parser.parse_args()
    for path in args.hdf5:
        meta, episodes = read_robomimic(path)
        dest = args.out / (meta.task_id or path.stem)
        n = write_lerobot(dest, meta, episodes, fps=args.fps, robot_type=args.robot_type)
        print(f"{path} -> {dest} ({n} episodes)")


if __name__ == "__main__":
    main()
