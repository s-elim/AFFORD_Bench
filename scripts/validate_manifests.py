"""Validate AFFORD-Bench benchmark manifests against JSON schemas and runtime invariants."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = ROOT / "schemas"
BENCHMARK_DIR = ROOT / "benchmark"


def main() -> None:
    manifest_schema_path = SCHEMAS_DIR / "manifest.schema.json"
    scenario_schema_path = SCHEMAS_DIR / "scenario.schema.json"

    manifest_schema = json.loads(manifest_schema_path.read_text(encoding="utf-8"))
    scenario_schema = json.loads(scenario_schema_path.read_text(encoding="utf-8"))

    # Register referencing schemas
    registry = jsonschema.Draft202012Validator.ID_OF(manifest_schema)
    resolver = jsonschema.RefResolver(
        base_uri=manifest_schema_path.as_uri(),
        referrer=manifest_schema,
        store={"scenario.schema.json": scenario_schema},
    )
    validator = jsonschema.Draft202012Validator(manifest_schema, resolver=resolver)

    manifest_files = [
        BENCHMARK_DIR / "manifests" / "afford_bench_full.json",
        BENCHMARK_DIR / "lite" / "afford_bench_lite.json",
    ]

    print("=" * 80)
    print("AFFORD-Bench Manifest Schema Validation")
    print("=" * 80)

    total_errors = 0
    for path in manifest_files:
        if not path.exists():
            print(f"[MISSING] {path}")
            total_errors += 1
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(data))
        if errors:
            print(f"[FAIL] {path.name}: {len(errors)} validation errors:")
            for err in errors[:5]:
                print(f"  - {err.message} at {list(err.path)}")
            total_errors += len(errors)
        else:
            cases = data.get("cases", [])
            case_ids = [c["case_id"] for c in cases]
            unique_ids = set(case_ids)
            dup_count = len(case_ids) - len(unique_ids)

            if dup_count > 0:
                print(f"[FAIL] {path.name}: contains {dup_count} duplicate case IDs!")
                total_errors += dup_count
            else:
                print(f"[PASS] {path.name}: {len(cases)} cases, all IDs unique, schema valid.")

    print("=" * 80)
    if total_errors == 0:
        print("ALL MANIFESTS PASSED VALIDATION!")
    else:
        print(f"VALIDATION FAILED WITH {total_errors} ERRORS.")
        sys.exit(1)


if __name__ == "__main__":
    main()
