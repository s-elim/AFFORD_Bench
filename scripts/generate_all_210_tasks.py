"""Batch oracle demonstration generator for all 210 AFFORD-Bench tasks (Robomimic HDF5).

Forwarding wrapper for scripts/generate_all_150_tasks.py (which covers all 210 core benchmark
tasks across domains D1 through D7, plus optional Domain 8 frontier extension tasks).

Usage:
  python scripts/generate_all_210_tasks.py --num-demos 50
  python scripts/generate_all_210_tasks.py --domain industrial --num-demos 50
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_all_150_tasks import main

if __name__ == "__main__":
    sys.exit(main())
