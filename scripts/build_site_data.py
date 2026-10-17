"""Build docs/site/data.js for the project page from the code and the evidence files (no hand-entered numbers).

Sources: task definitions and tiers (afford_pro), the runtime task audit, the sanity run (oracle, random, null on all
210 tasks), the leaderboard runs (scripts/evaluate_policy_suite.py on the core tier), the L4 friction x mass sweep, and
the dataset validation report. Every number on the page comes from one of these files; the file names are written into
the output so the page can cite them.

  python scripts/build_site_data.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "docs" / "roadmap" / "evidence"
SOURCES = {
    "task_audit": "task_audit_2026-10-17.json",
    "sanity": "sanity_2026-10-18",
    "leaderboard": "leaderboard_*",
    "l4_sweep": "l4_sweep_2026-10-18.csv",
    "validation": "dataset_validation_2026-10-17",
}
DOMAIN_SHORT = {
    "Industrial & Manufacturing": "D1 Industrial", "Culinary & Kitchen Operations": "D2 Culinary",
    "Tabletop & Domestic": "D3 Tabletop", "Laboratory & Life Sciences": "D4 Laboratory",
    "Long-Horizon Multi-Stage": "D5 Long-horizon", "Multi-Stage Precision Chaining": "D6 Precision chains",
    "Sequential Service & Culinary Assembly": "D7 Service chains",
}
SUITES = [
    ("nominal", "L1 Nominal", "Start of the evaluated body U(±7 cm), yaw U(±45°); TCP jitter ±1.5 cm"),
    ("p2_spatial", "L2 Spatial", "Start offset 7–12 cm in a random direction, |yaw| 45–90°"),
    ("p3_visual", "L3 Visual", "Light position ±0.5 m, diffuse ×0.5–1.5 (clipped at 1)"),
    ("p4_physical", "L4 Physical", "Friction μ 0.12–0.25 on body and finger pads; mass and inertia ×1.8–2.5"),
    ("combined_ood", "L5 Compound", "L2 + L3 + L4 + color"),
    ("p1_object", "Color", "Color of the evaluated body ×0.75–1.25"),
]
TRUE = {"True", "true", "1"}


def _csv_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _rate_block(per_task: dict[str, list[bool]]) -> dict[str, Any]:
    from afford_pro.evaluation.stats import summarize

    s = summarize(list(per_task.values()))
    return {"rate": s["rate"], "ci": s["task_cluster_bootstrap95"], "wilson": s["wilson95"],
            "episodes": s["episodes"], "tasks": s["tasks"]}


def tasks_block() -> list[dict[str, Any]]:
    from afford_pro.task_definitions import ALL_210_TASKS
    from afford_pro.task_tiers import NEAR_FIXED_START_CORE, UNMARKED_GOAL_CORE, mismatch_reason, task_tier

    audit = {r["task_id"]: r for r in json.loads((EVIDENCE / SOURCES["task_audit"]).read_text())["tasks"]}
    sanity = defaultdict(lambda: defaultdict(list))
    for e in _csv_rows(sorted((EVIDENCE / SOURCES["sanity"]).glob("episodes_*.csv"))):
        sanity[e["task_id"]][e["mode"]].append(e["success"] in TRUE and int(e["steps"]) <= 400)
    out = []
    for t in ALL_210_TASKS:
        a = audit.get(t.task_id, {})
        nominal, physical = sanity[t.task_id]["nominal"], sanity[t.task_id]["p4_physical"]
        out.append({
            "n": t.task_number, "id": t.task_id, "domain": DOMAIN_SHORT[t.domain], "tier": task_tier(t.task_id),
            "type": t.task_type, "object": t.object_name, "tol_mm": round(1000 * t.tolerance_xy, 1),
            "tilt_deg": t.tilt_tol_deg, "instruction": t.instruction, "reason": mismatch_reason(t.task_id),
            "goal_on": a.get("body_below_goal"), "hold_goal": bool(a.get("hold_goal", False)),
            "start_std_mm": round(1000 * max(a["start_xy_std_m"]), 1) if a.get("start_xy_std_m") else None,
            "unmarked_goal": t.task_id in UNMARKED_GOAL_CORE, "fixed_start": t.task_id in NEAR_FIXED_START_CORE,
            "oracle_nominal": [sum(nominal), len(nominal)], "oracle_l4": [sum(physical), len(physical)],
        })
    return out


def _method_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Per method (policy name before ':') and suite: pooled rate with intervals, plus mean and s.d. over the method's
    checkpoints (training seeds); reference policies have a single 'checkpoint'."""
    pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_ckpt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        method = r["policy"].split(":")[0]
        ok = r["success"] in TRUE
        pooled[method][r["suite"]][r["task_id"]].append(ok)
        per_ckpt[method][r["suite"]][r["policy"]].append(ok)
    out: dict[str, Any] = {}
    for method, suites in pooled.items():
        out[method] = {}
        for suite, tasks in suites.items():
            block = _rate_block(tasks)
            rates = [sum(v) / len(v) for v in per_ckpt[method][suite].values()]
            block["seeds"] = len(rates)
            block["seed_mean"] = sum(rates) / len(rates)
            block["seed_sd"] = (sum((x - block["seed_mean"]) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 if len(rates) > 1 else None
            out[method][suite] = block
    return out


def leaderboard_block() -> dict[str, Any]:
    paths = sorted(p for d in sorted(EVIDENCE.glob(SOURCES["leaderboard"])) for p in d.glob("*/episodes.csv"))
    core = _method_rows(_csv_rows(paths))
    sanity_tasks = _csv_rows(sorted((EVIDENCE / SOURCES["sanity"]).glob("part_*.csv")))
    sanity_eps = _csv_rows(sorted((EVIDENCE / SOURCES["sanity"]).glob("episodes_*.csv")))
    oracle = defaultdict(lambda: defaultdict(list))
    for e in sanity_eps:
        oracle[e["mode"]][e["task_id"]].append(e["success"] in TRUE and int(e["steps"]) <= 400)
    random_all = {t["task_id"]: [False] * 5 if float(t["random_sr"]) == 0 else
                  [True] * round(5 * float(t["random_sr"])) + [False] * (5 - round(5 * float(t["random_sr"])))
                  for t in sanity_tasks}
    null_all = {t["task_id"]: [float(t["null_sr"]) > 0] for t in sanity_tasks}
    all_tasks = {"oracle": {"nominal": _rate_block(oracle["nominal"]), "p4_physical": _rate_block(oracle["p4_physical"])},
                 "random": {"nominal": _rate_block(random_all)}, "null": {"nominal": _rate_block(null_all)}}
    return {"core": core, "all": all_tasks}


def sweep_block() -> list[dict[str, Any]]:
    cells = defaultdict(lambda: defaultdict(list))
    for r in _csv_rows([EVIDENCE / SOURCES["l4_sweep"]]):
        cells[(r["mu"], r["mass_scale"])][r["task_id"]].append(r["success"] in TRUE)
    return [{"mu": mu, "mass": float(mass), **_rate_block(tasks)} for (mu, mass), tasks in sorted(cells.items())]


def dataset_block() -> dict[str, Any]:
    report = json.loads((EVIDENCE / SOURCES["validation"] / "report.json").read_text())
    files = report["files"]
    counts: dict[str, int] = defaultdict(int)
    for f in files:
        for k, v in f.get("issue_counts", {}).items():
            counts[k] += v
    replays = [r for f in files for r in f.get("replay", [])]
    return {"files": len(files), "demos": sum(f["n_demos"] for f in files), "steps": sum(f["total_steps"] for f in files),
            "bytes": sum(f["size_bytes"] for f in files),
            "status": {s: sum(f["status"] == s for f in files) for s in ("VALID", "VALID_WITH_WARNINGS", "INVALID")},
            "issues": dict(sorted(counts.items())), "replays": len(replays),
            "replays_exact_step": sum(r["consistent"] for r in replays)}


def main() -> None:
    from afford_pro.provenance import collect_provenance

    prov = collect_provenance()
    data = {
        "generated": prov["timestamp_utc"], "commit": prov["git"]["commit"], "dirty": prov["git"]["dirty"],
        "mujoco": prov["packages"]["mujoco"], "sources": SOURCES,
        "suites": [{"key": k, "name": n, "desc": d} for k, n, d in SUITES],
        "tasks": tasks_block(), "leaderboard": leaderboard_block(), "l4_sweep": sweep_block(),
        "dataset": dataset_block(),
    }
    out = ROOT / "docs" / "site" / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.AFFORD = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(data['tasks'])} tasks, "
          f"{len(data['leaderboard']['core'])} leaderboard policies, {len(data['l4_sweep'])} sweep cells")


if __name__ == "__main__":
    main()
