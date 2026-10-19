"""Run provenance: code version, dependency versions, and content hashes of the assets a run depends on.

Every dataset, validation report, and evaluation result records `collect_provenance()` so that a number can be
traced to the exact code, simulator build, and scene files that produced it. Nothing here is inferred: fields
that cannot be determined are stored as None.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
ASSET_DIR = PACKAGE_DIR / "assets"
TRACKED_PACKAGES = ("mujoco", "numpy", "h5py", "gymnasium", "torch", "scipy")
# Source files whose content defines task semantics, physics, and the oracle.
SEMANTIC_SOURCES = (
    "task_definitions.py",
    "registry.py",
    "envs/base_franka_env.py",
    "envs/parametric_env.py",
    "envs/frontier_env.py",
    "oracles/base_oracle.py",
    "oracles/parametric_oracle.py",
    "oracles/frontier_oracle.py",
    "oracles/twist_lock_oracle.py",
    "perturbations/engine.py",
    "datasets/hdf5_writer.py",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_raw(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def _git(*args: str) -> str | None:
    out = _git_raw(*args)
    return None if out is None else out.strip()


def git_state() -> dict[str, Any]:
    """Commit, branch, and whether tracked files differ from the commit (a dirty tree cannot be re-checked out)."""
    # Unstripped: porcelain lines start with a two-letter status that may begin with a space.
    status = _git_raw("status", "--porcelain", "--untracked-files=no")
    modified = sorted(line[3:] for line in status.splitlines() if line.strip()) if status else []
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(modified),
        "modified_tracked_files": modified,
    }


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def asset_hashes(asset_dir: Path = ASSET_DIR) -> dict[str, str]:
    """sha256 of every scene XML and one combined digest over all mesh/texture files."""
    hashes = {p.name: sha256_file(p) for p in sorted(asset_dir.glob("*.xml"))}
    mesh_digest = hashlib.sha256()
    for p in sorted(q for q in asset_dir.rglob("*") if q.is_file() and q.suffix.lower() != ".xml"):
        mesh_digest.update(p.relative_to(asset_dir).as_posix().encode("utf-8"))
        mesh_digest.update(sha256_file(p).encode("ascii"))
    hashes["__non_xml_assets__"] = mesh_digest.hexdigest()
    return hashes


def source_hashes() -> dict[str, str | None]:
    return {rel: (sha256_file(PACKAGE_DIR / rel) if (PACKAGE_DIR / rel).exists() else None) for rel in SEMANTIC_SOURCES}


def collect_provenance(command: list[str] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to tell whether two runs used the same code, simulator, and scenes."""
    record: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": list(sys.argv if command is None else command),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": git_state(),
        "asset_sha256": asset_hashes(),
        "source_sha256": source_hashes(),
    }
    if extra:
        record.update(extra)
    return record


def provenance_json(command: list[str] | None = None, extra: dict[str, Any] | None = None) -> str:
    return json.dumps(collect_provenance(command, extra), sort_keys=True)
