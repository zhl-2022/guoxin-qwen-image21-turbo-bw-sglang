#!/usr/bin/env python3
"""Exact publisher Wan snapshot from the byte-matched ModelScope source lock.

Plan/verify are CPU-only; download runs only in the fixed SCNet namespace. No pip
or model/framework imports. Shared transfer code preserves owned resumable parts.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from download_candidate_models_20261023 import check_relative, download_file, json_new, verified_file

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
REVISION = "b184e23a8a16b20f108f727c902e769e873ffc73"
NAMESPACE = "Wan2.1-I2V-14B-480P-Diffusers"
BASE = Path("/root/private_data/guoxin-candidates/models")
CHECKPOINT = ROOT / "tasks/wan21-i2v14b480p-bw-sglang/checkpoint.lock.json"
SOURCES = ROOT / "tasks/wan21-i2v14b480p-bw-sglang/download-sources-006.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def no_links(path):
    if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
        raise ValueError("Symlink destination or ancestor refused")


def validate(source_lock, checkpoint_lock):
    if (checkpoint_lock.get("model_id"), checkpoint_lock.get("revision")) != (MODEL, REVISION):
        raise ValueError("Pinned HF model identity differs")
    if (source_lock.get("schema") != "session006-wan-fixed-transfer-v1"
            or source_lock.get("modelscope_complete") is not True
            or source_lock.get("all_36_published_size_SHA256_equal") is not True):
        raise ValueError("Incomplete source identity mapping")
    required = {x["path"]: x for x in checkpoint_lock["files"]}
    rows = source_lock["files"]
    if len(rows) != 36 or len(required) != 36 or len({x["path"] for x in rows}) != 36 or set(required) != {x["path"] for x in rows}:
        raise ValueError("Exact complete 36-file source set differs")
    for row in rows:
        check_relative(row["path"])
        expected = required[row["path"]]
        if (row["namespace"], row["original_model"], row["original_revision"], row["modelscope_model"]) != (NAMESPACE, MODEL, REVISION, MODEL):
            raise ValueError("Source identity or destination namespace differs")
        if (row["size"], row["sha256"]) != (expected["bytes"], expected["sha256"]):
            raise ValueError("Source size/SHA differs from the fixed checkpoint")
        if not re.fullmatch("[0-9a-f]{40}", row["modelscope_revision"]) or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid immutable revision or full SHA256")
        if row["download_url"] != f"https://modelscope.cn/models/{MODEL}/resolve/{row['modelscope_revision']}/{row['path']}":
            raise ValueError("Source URL differs")
    return rows


def apply(rows, max_seconds, verify_only=False):
    if sys.platform != "linux":
        raise ValueError("Actual transfer/verification requires SCNet Linux")
    destination = BASE / NAMESPACE
    no_links(destination)
    BASE.mkdir(parents=True, exist_ok=True)
    needed = sum(row["size"] for row in rows if not (destination / row["path"]).is_file())
    if not verify_only and shutil.disk_usage(BASE).free < needed + 1024**3:
        raise ValueError("Insufficient model storage and reserve")
    run_id = uuid.uuid4().hex
    lock_path = BASE / "wan006-transfer.lock"
    json_new(lock_path, {"run_id": run_id, "pid": os.getpid()})
    report = {"schema": "wan-checkpoint-receipt006-v1", "run_id": run_id,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "failed",
              "mode": "verify_only" if verify_only else "download", "destination": str(destination),
              "model": MODEL, "revision": REVISION, "files": [], "GPU_imported": False}
    deadline = time.monotonic() + max_seconds
    try:
        for row in sorted(rows, key=lambda x: x["size"]):
            target = destination / row["path"]
            no_links(target)
            if verify_only:
                if not verified_file(target, row, deadline):
                    raise ValueError("Checkpoint identity differs: " + row["path"])
                result = {"path": row["path"], "status": "verified", "size": row["size"], "sha256": row["sha256"]}
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                result = download_file(row, target, row["download_url"], deadline)
            report["files"].append({**result, "path": row["path"], "source": row["download_url"]})
            print(json.dumps({"event": "file_verified", **report["files"][-1]}), flush=True)
        report["status"] = "completed_all_36_files_verified"
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        json_new(BASE / ("wan006-transfer-" + run_id + ".json"), report)
        if lock_path.is_file() and read(lock_path).get("run_id") == run_id:
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download", "verify"))
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    rows = validate(read(args.sources), read(args.checkpoint))
    if args.action == "plan":
        print(json.dumps({"files": 36, "total_bytes": sum(x["size"] for x in rows),
                          "destination": str(BASE / NAMESPACE), "GPU_imported": False,
                          "network_executed": False}))
        return 0
    if not 60 <= args.max_seconds <= 21600:
        parser.error("Deadline must be in 60..21600 seconds")
    return apply(rows, args.max_seconds, verify_only=args.action == "verify")


if __name__ == "__main__":
    raise SystemExit(main())
