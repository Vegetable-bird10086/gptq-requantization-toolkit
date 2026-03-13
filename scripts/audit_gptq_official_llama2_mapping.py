#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def load_gptq_module_names(gptq_encodings_path: Path) -> list[str]:
    with open(gptq_encodings_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("param_encodings", [])
    names: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    names.append(name)
    elif isinstance(raw, dict):
        names.extend([k for k in raw.keys() if isinstance(k, str)])

    return sorted(set(names))


def load_official_param_keys(official_encodings_path: Path) -> set[str]:
    with open(official_encodings_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    param_enc = data.get("param_encodings", {})
    if not isinstance(param_enc, dict):
        raise RuntimeError("Official encodings param_encodings must be dict format.")
    return {k for k in param_enc.keys() if isinstance(k, str)}


def candidate_exact_names(module_name: str) -> list[str]:
    return [
        module_name,
        f"{module_name}.weight",
        module_name.replace(".", "/"),
        f"{module_name.replace('.', '/')}.weight",
    ]


def structural_targets(module_name: str, available: set[str]) -> list[str]:
    m = re.match(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(.+)$", module_name)
    if not m:
        return []

    layer_idx = m.group(1)
    block = m.group(2)
    proj = m.group(3)
    prefix = f"model.model.model.layers.{layer_idx}."

    if block == "self_attn" and proj in {"q_proj", "k_proj", "v_proj"}:
        base = f"{prefix}self_attn.{proj}_sha."
        keys = sorted([k for k in available if k.startswith(base) and k.endswith(".weight")])
        return keys

    if block == "self_attn" and proj == "o_proj":
        key = f"{prefix}self_attn.o_proj_conv.weight"
        return [key] if key in available else []

    if block == "mlp" and proj in {"gate_proj", "up_proj", "down_proj"}:
        key = f"{prefix}mlp.{proj}_conv.weight"
        return [key] if key in available else []

    return []


def audit_one(official_path: Path, gptq_modules: list[str]) -> dict:
    available = load_official_param_keys(official_path)

    exact_hits = 0
    exact_hit_modules: list[str] = []
    structural_hits = 0
    unmatched_modules: list[str] = []
    mapped_key_union: set[str] = set()
    module_to_targets: dict[str, list[str]] = {}

    for name in gptq_modules:
        exact_name = next((c for c in candidate_exact_names(name) if c in available), None)
        if exact_name is not None:
            exact_hits += 1
            exact_hit_modules.append(name)
            mapped_key_union.add(exact_name)
            module_to_targets[name] = [exact_name]
            continue

        targets = structural_targets(name, available)
        if targets:
            structural_hits += 1
            mapped_key_union.update(targets)
            module_to_targets[name] = targets
        else:
            unmatched_modules.append(name)

    total_gptq = len(gptq_modules)
    total_official = len(available)

    return {
        "official_file": str(official_path),
        "total_gptq_modules": total_gptq,
        "total_official_param_keys": total_official,
        "exact_match_modules": exact_hits,
        "structural_match_modules": structural_hits,
        "unmatched_modules": len(unmatched_modules),
        "official_keys_covered": len(mapped_key_union),
        "official_coverage_ratio": (len(mapped_key_union) / total_official) if total_official else 0.0,
        "unmatched_module_names": unmatched_modules,
        "exact_hit_module_names": exact_hit_modules,
        "module_to_targets": module_to_targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit GPTQ module names against official Llama2 llama_sha encodings using exact and structural mapping."
    )
    parser.add_argument("--gptq-encodings", required=True, type=Path)
    parser.add_argument(
        "--official-encodings",
        required=True,
        nargs="+",
        help="One or more official encodings paths or glob patterns.",
    )
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args()

    gptq_modules = load_gptq_module_names(args.gptq_encodings)

    official_paths: list[Path] = []
    for item in args.official_encodings:
        matches = sorted(glob.glob(item))
        if matches:
            official_paths.extend([Path(m) for m in matches])
        else:
            official_paths.append(Path(item))

    official_paths = [p for p in official_paths if p.exists()]
    if not official_paths:
        raise RuntimeError("No official encodings files found from --official-encodings.")

    reports = [audit_one(p, gptq_modules) for p in official_paths]

    summary = {
        "gptq_encodings": str(args.gptq_encodings),
        "gptq_module_count": len(gptq_modules),
        "files": reports,
    }

    if args.out_report:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    for r in reports:
        print(Path(r["official_file"]).name)
        print(f"  total_gptq_modules: {r['total_gptq_modules']}")
        print(f"  exact_match_modules: {r['exact_match_modules']}")
        print(f"  structural_match_modules: {r['structural_match_modules']}")
        print(f"  unmatched_modules: {r['unmatched_modules']}")
        print(f"  official_keys_covered: {r['official_keys_covered']} / {r['total_official_param_keys']}")
        print(f"  official_coverage_ratio: {r['official_coverage_ratio']:.4f}")

    if args.out_report:
        print(f"report: {args.out_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
