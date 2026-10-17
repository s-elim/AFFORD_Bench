"""Validate every episode of every AFFORD-Bench dataset file and write machine- and human-readable reports.

Usage:
  python scripts/validate_datasets.py --out-dir docs/roadmap/evidence/dataset_validation
  python scripts/validate_datasets.py --replay-per-file 2 --workers 8          # also replay 2 demos per file
  python scripts/validate_datasets.py --files datasets/tabletop_sort_red_nominal.hdf5 --image-frames all

Outputs in --out-dir: report.json (provenance + one summary per file), summary.md, manifest.json (file, size,
status, optional sha256). Exit code 1 if any file is INVALID.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_frames(text: str) -> tuple[int, ...] | str:
    if text == "none":
        return ()
    if text == "all":
        return "all"
    return tuple(int(x) for x in text.split(","))


def _replay_sample(keys: list[str], n: int) -> list[str]:
    """First, last, and evenly spaced demos, so a file's replay covers its whole seed range."""
    if n <= 0 or not keys:
        return []
    idx = np.unique(np.linspace(0, len(keys) - 1, num=min(n, len(keys))).round().astype(int))
    return [keys[i] for i in idx]


def check_one(path: str, expected_demos: int, frames: Any, replay_n: int, sha: bool) -> dict[str, Any]:
    """Validate one file; any failure becomes an INVALID row rather than aborting the whole run."""
    try:
        return _check_one(path, expected_demos, frames, replay_n, sha)
    except Exception as exc:  # noqa: BLE001 - reported per file, never swallowed
        task_id, mode = Path(path).stem, "unknown"
        return {"path": path, "task_id": task_id, "mode": mode, "status": "INVALID", "n_demos": 0, "total_steps": 0,
                "stats": {}, "issue_counts": {f"error:validator_crash:{type(exc).__name__}": 1},
                "first_issues": [{"level": "error", "code": "validator_crash", "message": repr(exc), "demo": None}],
                "size_bytes": Path(path).stat().st_size if Path(path).exists() else 0, "sha256": None}


def _longest_episode(p: Path) -> int:
    import h5py
    with h5py.File(p, "r") as f:
        lengths = [f["data"][k]["actions"].shape[0] for k in f.get("data", {}) if k.startswith("demo_")]
    return max(lengths, default=0)


def _check_one(path: str, expected_demos: int, frames: Any, replay_n: int, sha: bool) -> dict[str, Any]:
    from afford_pro.datasets.replay import replay_file
    from afford_pro.datasets.validation import build_task_context, parse_dataset_name, validate_file
    from afford_pro.provenance import sha256_file

    p = Path(path)
    ctx = build_task_context(*parse_dataset_name(p))
    if frames == "all":
        frames = tuple(range(_longest_episode(p)))  # validate_file keeps only frames inside each episode
    report = validate_file(p, expected_demos, frames, ctx)
    out = report.summary()
    if replay_n and report.status != "INVALID" and ctx.instruction is not None:
        keys = [f"demo_{i}" for i in range(report.n_demos)]
        rows = replay_file(p, _replay_sample(keys, replay_n), ctx.qpos_offset, ctx.object_qpos_adr, ctx.arm_qpos_adr)
        out["replay"] = rows
        # Files from the current generator (they carry provenance) execute exactly the actions they record, so their
        # replays must be exact; legacy files stored float32 roundings of the executed actions (AUDIT_REPORT A-26).
        strict = "warning:no_provenance" not in out.get("issue_counts", {})
        bad = [r["demo"] for r in rows if not r["consistent"]
               or (strict and (r["max_object_dev_m"] > 0 or r["max_joint_dev_rad"] > 0))]
        if bad and strict:
            out.setdefault("issue_counts", {})["error:replay_mismatch"] = len(bad)
            out["status"] = "INVALID"
        elif bad:
            out.setdefault("issue_counts", {})["warning:replay_inconsistent"] = len(bad)
            out["status"] = "VALID_WITH_WARNINGS" if out["status"] == "VALID" else out["status"]
    out["size_bytes"] = p.stat().st_size
    out["sha256"] = sha256_file(p) if sha else None
    return out


def _cross_file_duplicates(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for h in set(r["stats"].get("action_hashes", [])):
            owners[h].append(r["task_id"])
    return {h: sorted(set(t)) for h, t in owners.items() if len(set(t)) > 1}


def _summary_md(results: list[dict[str, Any]], dups: dict[str, list[str]]) -> str:
    status = Counter(r["status"] for r in results)
    codes: Counter = Counter()
    for r in results:
        codes.update({k: v for k, v in r.get("issue_counts", {}).items()})
    files_with: Counter = Counter()
    for r in results:
        files_with.update(r.get("issue_counts", {}).keys())
    demos = sum(r["n_demos"] for r in results)
    steps = sum(r["total_steps"] for r in results)
    lines = ["# Dataset validation summary", "",
             f"Files: {len(results)}; demos: {demos}; steps: {steps}; mean steps per demo: {steps / max(demos, 1):.1f}", "",
             "| Status | Files |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in sorted(status.items())]
    lines += ["", "| Issue (level:code) | Occurrences | Files |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v} | {files_with[k]} |" for k, v in sorted(codes.items())]
    replays = [row for r in results for row in r.get("replay", [])]
    if replays:
        ok = sum(row["consistent"] for row in replays)
        lines += ["", f"Replay: {ok}/{len(replays)} demos reach success exactly at their last step under the current code; "
                  f"max object deviation {max(row['max_object_dev_m'] for row in replays):.4f} m (median "
                  f"{np.median([row['max_object_dev_m'] for row in replays]):.5f} m)."]
    lines += ["", f"Action sequences shared across tasks: {len(dups)}"]
    invalid = [r for r in results if r["status"] == "INVALID"]
    if invalid:
        lines += ["", "## INVALID files", ""] + [f"- {Path(r['path']).name}: {r['issue_counts']}" for r in invalid]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--pattern", default="*_nominal.hdf5")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--expected-demos", type=int, default=50)
    parser.add_argument("--image-frames", default="0,-1", help="comma list of frame indices, 'all', or 'none'")
    parser.add_argument("--replay-per-file", type=int, default=0)
    parser.add_argument("--sha256", action="store_true", help="hash every file (slow for hundreds of GB)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "roadmap" / "evidence" / "dataset_validation")
    args = parser.parse_args()

    from afford_pro.provenance import collect_provenance

    files = [Path(f) for f in args.files] if args.files else sorted(args.data_dir.glob(args.pattern))
    frames = _parse_frames(args.image_frames)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_one, str(p), args.expected_demos, frames, args.replay_per_file, args.sha256)
                   for p in files]
        results = []
        for p, fut in zip(files, futures):
            res = fut.result()
            results.append(res)
            print(f"{res['status']:20s} {p.name} {res.get('issue_counts', {})}", flush=True)

    dups = _cross_file_duplicates(results)
    for r in results:
        r["stats"].pop("action_hashes", None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"provenance": collect_provenance(), "settings": {k: str(v) for k, v in vars(args).items()},
              "cross_file_duplicate_action_sequences": dups, "files": results}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    manifest = [{"file": Path(r["path"]).name, "task_id": r["task_id"], "mode": r["mode"], "status": r["status"],
                 "n_demos": r["n_demos"], "total_steps": r["total_steps"], "size_bytes": r["size_bytes"],
                 "sha256": r["sha256"]} for r in results]
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(_summary_md(results, dups), encoding="utf-8")
    print(f"Wrote {args.out_dir}")
    return int(any(r["status"] == "INVALID" for r in results))


if __name__ == "__main__":
    sys.exit(main())
