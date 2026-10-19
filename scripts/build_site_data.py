"""Build docs/site/data.js for the project page from the code and the evidence files (no hand-entered numbers).

Sources: the task specifications (afford_pro.bench), the task validation runs (scripts/validate_bench_tasks.py) and the
leaderboard runs (scripts/evaluate_policy_suite.py, one folder per run with episodes.csv and summary.json). Every number
on the page comes from one of these files; their names are written into the output so the page can cite them.

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
SOURCES = {"tasks": "afford_pro/bench/tasks", "validation": "task_validation_*_<suite>.json (newest per suite)",
           "leaderboard": "leaderboard_*/*/summary.json (task_set bench)"}
VALIDATION = "task_validation_*_{}.json"   # newest file per suite
DOMAIN_SHORT = {
    "Industrial & Manufacturing": "D1 Industrial", "Culinary & Kitchen Operations": "D2 Culinary",
    "Tabletop & Domestic": "D3 Tabletop", "Laboratory & Life Sciences": "D4 Laboratory",
    "Long-Horizon Multi-Stage": "D5 Long-horizon", "Multi-Stage Precision Chaining": "D6 Precision chains",
    "Sequential Service & Culinary Assembly": "D7 Service chains",
}
WORKCELL = {"industrial_assembly": "Assembly jig", "industrial_electronics": "Electronics bench",
            "kitchen_burger": "Burger station", "kitchen_coffee": "Espresso station",
            "tabletop_sorting": "Sorting table", "lab_science": "Laboratory bench"}
SUITES = [
    ("nominal", "L1 Nominal", "Each moved object starts uniformly in its own region (yaw up to ±45°), in one of two receptacles, or pre-assembled; TCP jitter ±1.5 cm"),
    ("p2_spatial", "L2 Spatial", "Region starts offset 7–12 cm in a random direction, |yaw| 45–90°"),
    ("p3_visual", "L3 Visual", "Light position ±0.5 m, diffuse ×0.5–1.5 (clipped at 1)"),
    ("p4_physical", "L4 Physical", "Friction μ 0.12–0.25 on the moved objects and finger pads; their mass and inertia ×1.8–2.5"),
    ("combined_ood", "L5 Compound", "L2 + L3 + L4 + color"),
    ("p1_object", "Color", "Color of the moved objects ×0.75–1.25"),
]
VALIDATED_SUITES = ("nominal", "p4_physical", "p2_spatial", "combined_ood")
TRUE = {"True", "true", "1"}


def _rate_block(per_task: dict[str, list[bool]]) -> dict[str, Any]:
    from afford_pro.evaluation.stats import summarize

    s = summarize(list(per_task.values()))
    return {"rate": s["rate"], "ci": s["task_cluster_bootstrap95"], "wilson": s["wilson95"],
            "episodes": s["episodes"], "tasks": s["tasks"]}


def _validation() -> dict[str, dict[str, Any]]:
    out = {}
    for suite in VALIDATED_SUITES:
        paths = sorted(EVIDENCE.glob(VALIDATION.format(suite)))
        if not paths:
            print(f"warning: no validation run for suite {suite} ({VALIDATION.format(suite)})", file=sys.stderr)
            continue
        out[suite] = json.loads(paths[-1].read_text(encoding="utf-8"))
        out[suite]["source"] = paths[-1].name
    return out


def tasks_block(validation: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from afford_pro.bench.language import step_text
    from afford_pro.bench.scenes import SCENES
    from afford_pro.bench.spec import step_budget
    from afford_pro.bench.tasks import TASKS

    oracle = {suite: {r["number"]: [r["oracle_successes"], len(r["episodes"])] for r in run["tasks"] if "error" not in r}
              for suite, run in validation.items()}
    out = []
    for t in TASKS:
        tol_mm, tilt = SCENES[t.scene].precision
        out.append({
            "n": t.number, "id": t.task_id, "domain": DOMAIN_SHORT[t.domain], "workcell": WORKCELL[t.scene],
            "instruction": t.instruction, "goals": [step_text(g) for g in t.goals],
            "kinds": sorted({type(g).__name__ for g in t.goals}), "ordered": bool(t.order),
            "budget": step_budget(t), "tol_mm": round(1000 * tol_mm, 1), "tilt_deg": tilt,
            "oracle": {suite: rates.get(t.number) for suite, rates in oracle.items()},
        })
    return out


def _leaderboard_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in sorted(EVIDENCE.glob("leaderboard_*/*/summary.json")):
        settings = json.loads(summary.read_text(encoding="utf-8")).get("settings", {})
        if settings.get("task_set") != "bench":
            continue
        with (summary.parent / "episodes.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


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


def leaderboard_block(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Methods x suites over all tasks ("all") and per domain."""
    rows = _leaderboard_rows()
    domain_of = {t["id"]: t["domain"] for t in tasks}
    blocks = {"all": _method_rows(rows)}
    for domain in sorted(set(domain_of.values())):
        blocks[domain] = _method_rows([r for r in rows if domain_of.get(r["task_id"]) == domain])
    return blocks


def validation_block(validation: dict[str, dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Oracle success per suite and domain in the validation runs, plus the floors and start audit of the nominal run."""
    domain_of = {t["n"]: t["domain"] for t in tasks}
    ceiling: dict[str, dict[str, list[int]]] = {}
    per_task = {suite: run["episodes"] for suite, run in validation.items()}
    for suite, run in validation.items():
        cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in run["tasks"]:
            for key in ("All", domain_of[r["number"]]):
                cells[key][0] += r.get("oracle_successes", 0)
                cells[key][1] += len(r.get("episodes", []))
        ceiling[suite] = dict(cells)
    nominal = validation.get("nominal", {"tasks": []})
    built = [r for r in nominal["tasks"] if "error" not in r]  # a task that failed to build ran no floors or resets
    floors = defaultdict(int)
    for r in built:
        for k, v in r.get("floor_successes", {}).items():
            floors[k] += v
    return {"ceiling": ceiling, "episodes_per_task": per_task, "floors": dict(floors),
            "floor_episodes": nominal.get("floor_episodes", 0) * len(built), "resets": nominal.get("resets", 0) * len(built),
            "goal_true_at_reset": sum(r.get("goal_true_at_reset", 0) for r in built),
            "start_fallbacks": sum(r.get("start_fallbacks", 0) for r in built), "tasks_built": len(built),
            "tasks": len(nominal["tasks"]), "sources": {s: run.get("source") for s, run in validation.items()}}


def main() -> None:
    from afford_pro.provenance import collect_provenance

    prov = collect_provenance()
    validation = _validation()
    tasks = tasks_block(validation)
    data = {
        "generated": prov["timestamp_utc"], "commit": prov["git"]["commit"], "dirty": prov["git"]["dirty"],
        "mujoco": prov["packages"]["mujoco"], "sources": SOURCES,
        "suites": [{"key": k, "name": n, "desc": d} for k, n, d in SUITES],
        "tasks": tasks, "leaderboard": leaderboard_block(tasks), "validation": validation_block(validation, tasks),
    }
    out = ROOT / "docs" / "site" / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.AFFORD = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(tasks)} tasks, "
          f"{len(data['leaderboard']['all'])} leaderboard policies, {len(validation)} validation suites")


if __name__ == "__main__":
    main()
