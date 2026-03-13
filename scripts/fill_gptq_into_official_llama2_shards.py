#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fill GPTQ param encodings into official llama_sha_0..3 encodings with shard layer offsets "
            "(0, 8, 16, 24)."
        )
    )
    parser.add_argument("--gptq-dir", required=True, type=Path)
    parser.add_argument("--gptq-encodings", required=True, type=Path)
    parser.add_argument(
        "--official-config-dir",
        required=True,
        type=Path,
        help="Directory containing llama_sha_0.encodings ... llama_sha_3.encodings",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mapping-mode", type=str, default="structural", choices=["exact", "structural", "auto"])
    parser.add_argument("--clean", action="store_true", help="Remove out-dir before running.")
    args = parser.parse_args()

    merge_script = Path(__file__).parent / "merge_gptq_into_aimet_encodings.py"
    if not merge_script.exists():
        raise RuntimeError(f"merge script not found: {merge_script}")

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []

    for i in range(4):
        src_enc = args.official_config_dir / f"llama_sha_{i}.encodings"
        if not src_enc.exists():
            raise RuntimeError(f"missing official shard encodings: {src_enc}")

        ckpt_dir = args.out_dir / f"sha_{i}"
        out_ckpt = args.out_dir / f"sha_{i}_merged"

        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if out_ckpt.exists():
            shutil.rmtree(out_ckpt)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_enc, ckpt_dir / "model.encodings")

        cmd = [
            sys.executable,
            str(merge_script),
            "--gptq-dir",
            str(args.gptq_dir),
            "--gptq-encodings",
            str(args.gptq_encodings),
            "--aimet-checkpoint",
            str(ckpt_dir),
            "--out-checkpoint",
            str(out_ckpt),
            "--mapping-mode",
            args.mapping_mode,
            "--layer-offset",
            str(i * 8),
            "--device",
            args.device,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(f"merge failed for sha_{i}")

        report_path = out_ckpt / "gptq_merge_report.json"
        report = json.load(open(report_path, "r", encoding="utf-8"))
        summary.append(
            {
                "shard": i,
                "layer_offset": report.get("layer_offset"),
                "updated_entries": report.get("updated_entries"),
                "updated_target_keys": report.get("updated_target_keys"),
                "unmatched_modules": len(report.get("unmatched_modules", [])),
                "mapping_failures": len(report.get("mapping_failures", [])),
                "merged_model_encodings": str(out_ckpt / "model.encodings"),
            }
        )

    summary_path = args.out_dir / "fill_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for row in summary:
        print(
            f"sha_{row['shard']}: offset={row['layer_offset']} "
            f"updated_entries={row['updated_entries']} "
            f"updated_target_keys={row['updated_target_keys']} "
            f"unmatched={row['unmatched_modules']} "
            f"failures={row['mapping_failures']}"
        )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
