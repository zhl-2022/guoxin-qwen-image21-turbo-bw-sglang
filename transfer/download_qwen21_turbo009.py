"""Bounded independent Qwen21 transfer using the reviewed stdlib worker code."""
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
import threading
import time
import traceback
import uuid

from download_candidate_models_20261023 import check_relative, json_new
from download_wan_models_006_002 import run_workers, remaining_bytes, digest

MODEL = "Qwen/Qwen-Image-2.1-Turbo"
REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
LOCK_SHA256 = "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"
BASE = Path("/root/private_data/guoxin-candidates/models")
NAMESPACE = "Qwen-Image-2.1-Turbo-fc13ba1986f"
TRANSFER_LOCK = BASE / "qwen21-turbo009-transfer.lock"


def no_links(path):
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Symlink target or ancestor refused")


def validate(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCK_SHA256:
        raise ValueError("Pinned Qwen source lock bytes differ")
    lock = json.loads(raw)
    if (lock["model_id"], lock["hf_revision"]) != (MODEL, REVISION):
        raise ValueError("Pinned model/revision differs")
    rows = lock["files"]
    if len(rows) != 17 or len({row["path"] for row in rows}) != 17:
        raise ValueError("Require exact17 immutable source files")
    for row in rows:
        check_relative(row["path"])
        if not re.fullmatch("[0-9a-f]{64}", row["sha256"]) or not re.fullmatch("[0-9a-f]{40}", row["modelscope_revision"]):
            raise ValueError("Malformed source digest/revision")
        if row["download_url"] != f"https://modelscope.cn/models/{MODEL}/resolve/{row['modelscope_revision']}/{row['path']}":
            raise ValueError("Source URL differs")
    return rows


def select(rows, component):
    return [row for row in rows if component == "all" or row["component"] == "metadata"
            or (component == "vae" and row["component"] == "vae")]


def apply(rows, args):
    if sys.platform != "linux":
        raise ValueError("Actual transfer/verification is restricted to native SCNet Linux")
    if sum(row["size"] for row in rows) > args.max_bytes:
        raise ValueError("Selected phase exceeds explicit byte budget; complete32GB phase remains held by default")
    no_links(BASE); no_links(TRANSFER_LOCK)
    BASE.mkdir(parents=True, exist_ok=True)
    target = BASE / NAMESPACE
    no_links(target)
    run_id = uuid.uuid4().hex
    descriptor = os.open(TRANSFER_LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "pid": os.getpid(), "model": MODEL, "namespace": NAMESPACE}, stream)
    stop = threading.Event()
    record = {"schema": "qwen21-turbo009-transfer-receipt-v1", "run_id": run_id,
        "model": MODEL, "hf_revision": REVISION, "destination": str(target), "phase": args.component,
        "mode": args.action, "max_workers": 2, "status": "failed", "files": [],
        "GPU_imported": False, "full_checkpoint_verified": False,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "all_workers_exited_before_lock_release": False}
    deadline = time.monotonic() + args.max_seconds
    try:
        if args.action == "verify":
            for row in rows:
                path = target / row["path"]
                no_links(path)
                if not path.is_file() or path.stat().st_size != row["size"] or digest(path, deadline, stop) != row["sha256"]:
                    raise ValueError("Phase checkpoint differs: " + row["path"])
                record["files"].append({"path": row["path"], "status": "verified", "size": row["size"], "sha256": row["sha256"]})
            interrupted = False
            allowed = {"verified"}
        else:
            needed = remaining_bytes(rows, target)
            record["remaining_network_bytes"] = needed
            if shutil.disk_usage(BASE).free < needed + 1024**3:
                raise ValueError("Insufficient storage and reserve")
            # Only the shared per-file/worker functions are called. Wan apply(),
            # namespace and transfer lock are never used by this controller.
            ordered = sorted(rows, key=lambda row: ((target / row["path"]).is_file(), -row["size"]))
            record["files"], interrupted = run_workers(ordered, target, deadline, stop, 2)
            allowed = {"downloaded_verified", "reused_verified", "promoted_verified_partial"}
        record["all_workers_exited_before_lock_release"] = True
        passed = len(record["files"]) == len(rows) and all(row["status"] in allowed for row in record["files"])
        record["status"] = "completed_selected_phase_full_SHA_verified" if passed else "failed_partials_preserved"
        record["full_checkpoint_verified"] = passed and args.component == "all"
        if interrupted:
            record["status"] = "interrupted_all_workers_stopped_partials_preserved"
            return 130
        return 0 if passed else 2
    except BaseException as error:
        stop.set()
        record.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["torch_in_sys_modules"] = "torch" in sys.modules
        json_new(BASE / ("qwen21-turbo009-transfer-" + run_id + ".json"), record)
        if TRANSFER_LOCK.is_file() and json.loads(TRANSFER_LOCK.read_text()).get("run_id") == run_id:
            TRANSFER_LOCK.unlink()
        print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download", "verify"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().with_name("download-sources009.json"))
    parser.add_argument("--component", choices=("metadata", "vae", "all"), default="metadata")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-bytes", type=int, default=1024**3)
    args = parser.parse_args()
    if not 60 <= args.max_seconds <= 21600:
        parser.error("Deadline must be60..21600 seconds")
    rows = select(validate(args.lock), args.component)
    if args.action == "plan":
        print(json.dumps({"model": MODEL, "revision": REVISION, "phase": args.component,
            "files": len(rows), "bytes": sum(row["size"] for row in rows), "workers": 2,
            "destination": str(BASE / NAMESPACE), "lock": str(TRANSFER_LOCK),
            "network_executed": False, "GPU_imported": False,
            "full_checkpoint_phase_selected": args.component == "all"}, indent=2))
        return 0
    return apply(rows, args)


if __name__ == "__main__":
    raise SystemExit(main())
