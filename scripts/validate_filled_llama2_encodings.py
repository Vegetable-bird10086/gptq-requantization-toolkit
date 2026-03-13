#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


QMAX_4BIT = 15


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _entry_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def _check_min_max(scale: float, offset: int, mn: float, mx: float, atol: float = 1e-6) -> bool:
    exp_min = (0 - offset) * scale
    exp_max = (QMAX_4BIT - offset) * scale
    return math.isclose(mn, exp_min, abs_tol=atol) and math.isclose(mx, exp_max, abs_tol=atol)


def validate_shard(i: int, official_dir: Path, filled_dir: Path) -> dict:
    official_path = official_dir / f"llama_sha_{i}.encodings"
    merged_path = filled_dir / f"sha_{i}_merged" / "model.encodings"
    report_path = filled_dir / f"sha_{i}_merged" / "gptq_merge_report.json"

    off = _load(official_path)
    new = _load(merged_path)
    rpt = _load(report_path)

    off_keys = set(off["param_encodings"].keys())
    new_keys = set(new["param_encodings"].keys())

    if off_keys != new_keys:
        missing = sorted(off_keys - new_keys)
        extra = sorted(new_keys - off_keys)
        raise RuntimeError(
            f"sha_{i}: key mismatch: missing={len(missing)} extra={len(extra)}"
        )

    changed_keys = [k for k in sorted(off_keys) if new["param_encodings"][k] != off["param_encodings"][k]]

    length_mismatch = 0
    formula_fail = 0
    bitwidth_fail = 0

    for k in changed_keys:
        off_list = _entry_list(off["param_encodings"][k])
        new_list = _entry_list(new["param_encodings"][k])

        if len(off_list) != len(new_list):
            length_mismatch += 1
            continue

        for e in new_list:
            bw = int(e.get("bitwidth", -1))
            sc = float(e.get("scale", 0.0))
            of = int(e.get("offset", 0))
            mn = float(e.get("min", 0.0))
            mx = float(e.get("max", 0.0))
            if bw != 4:
                bitwidth_fail += 1
            if not _check_min_max(sc, of, mn, mx):
                formula_fail += 1

    unmatched = rpt.get("unmatched_modules", [])
    failures = rpt.get("mapping_failures", [])
    module_to_targets = rpt.get("module_to_targets", {})

    return {
        "shard": i,
        "official_keys": len(off_keys),
        "changed_keys": len(changed_keys),
        "updated_entries": rpt.get("updated_entries", -1),
        "updated_target_keys": rpt.get("updated_target_keys", -1),
        "unmatched_modules": len(unmatched),
        "mapping_failures": len(failures),
        "mapped_modules": len(module_to_targets),
        "length_mismatch": length_mismatch,
        "bitwidth_fail": bitwidth_fail,
        "formula_fail": formula_fail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate filled llama_sha encodings consistency.")
    parser.add_argument("--official-config-dir", required=True, type=Path)
    parser.add_argument("--filled-dir", required=True, type=Path)
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args()

    rows = [validate_shard(i, args.official_config_dir, args.filled_dir) for i in range(4)]

    mapped_union = set()
    for i in range(4):
        rpt_path = args.filled_dir / f"sha_{i}_merged" / "gptq_merge_report.json"
        rpt = _load(rpt_path)
        mapped_union.update(rpt.get("module_to_targets", {}).keys())

    summary = {
        "shards": rows,
        "mapped_modules_union": len(mapped_union),
        "all_ok": all(
            r["mapping_failures"] == 0
            and r["length_mismatch"] == 0
            and r["bitwidth_fail"] == 0
            and r["formula_fail"] == 0
            for r in rows
        ),
    }

    for r in rows:
        print(
            f"sha_{r['shard']}: keys={r['official_keys']} changed={r['changed_keys']} "
            f"updated_target_keys={r['updated_target_keys']} mapped_modules={r['mapped_modules']} "
            f"unmatched={r['unmatched_modules']} failures={r['mapping_failures']} "
            f"len_fail={r['length_mismatch']} bw_fail={r['bitwidth_fail']} formula_fail={r['formula_fail']}"
        )
    print(f"mapped_modules_union: {summary['mapped_modules_union']}")
    print(f"all_ok: {summary['all_ok']}")

    if args.out_report:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"report: {args.out_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
