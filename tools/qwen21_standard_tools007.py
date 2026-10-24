"""Read-only standard environment, frozen-example/result smoke and formal gates."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
from audit_qwen21_full014_actual import complete_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("environment", "smoke", "verify", "bench"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.dry_run:
        print(json.dumps({"action": args.action, "scope": "plan only", "GPU_model_executed": False}))
        return 0 if args.action in ("environment", "smoke") else 2
    if args.action == "environment":
        record = {"UTC": dt.datetime.now(dt.timezone.utc).isoformat(), "executable": sys.executable,
            "sys_version": sys.version, "packages": {}, "image_complete_OCI_reference": None}
        for name in ("torch", "transformers", "diffusers", "triton", "sglang", "sglang-kernel", "Pillow"):
            try:
                dist = metadata.distribution(name)
                record["packages"][name] = {"version": dist.version, "location": str(dist.locate_file(""))}
            except metadata.PackageNotFoundError:
                record["packages"][name] = None
        import torch
        record.update(torch_runtime_version=torch.__version__, hip=torch.version.hip,
            cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            record["device_name"] = torch.cuda.get_device_name(0)
            record["GPU_free_total_bytes"] = list(torch.cuda.mem_get_info(0))
        print(json.dumps(record, indent=2))
        return 0
    if args.action == "smoke":
        if args.results is None: parser.error("--results requires an actual completed result directory")
        receipt = args.results / "complete-pretrained014.json"
        record = json.loads(receipt.read_text())
        name = "actual/complete-pretrained014.json"
        contents = {"actual/" + row["png"]: (args.results / row["png"]).read_bytes() for row in record["samples"]}
        lock = json.loads((root / "transfer/download-sources009.json").read_text())
        expected = {row["path"]: row for row in lock["files"]}
        source_lock = json.loads((root / "core/source-lock.json").read_text())
        result = complete_contract(record, name, contents, expected, source_lock, root / "pretrained")
        assert result["passed"] is True
        result["scope"] = "CPU read-back of existing actual output; no new generation or HTTP call"
        print(json.dumps(result, indent=2))
        return 0
    required = ["QWEN_OFFICIAL_" + args.action.upper() + "_SCRIPT",
        "QWEN_OFFICIAL_DATA_MANIFEST", "QWEN_OFFICIAL_BASELINE", "QWEN_OFFICIAL_SCORING_CONTRACT"]
    print(json.dumps({"status": "blocked_official_material_contract_not_integrated",
        "missing_environment": [name for name in required if not os.environ.get(name)],
        "required_materials": required, "official_evaluation_executed": False,
        "official_accuracy_verified": False, "official_performance_verified": False,
        "submitted": False,
        "reason": "Bind the actual organizer evaluator/self-test/data/baseline/scoring interface before implementing a formal adapter; a development PNG or returncode does not establish an official score"}, indent=2))
    return 2


if __name__ == "__main__": raise SystemExit(main())
