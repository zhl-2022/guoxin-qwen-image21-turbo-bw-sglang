#!/usr/bin/env python3
"""Bounded three-file Wan transfer; preserves v001 owned partials and namespace.

Plan is offline. Download starts only after the old transfer has exited and its
namespace lock is absent. This program never removes or takes over another lock.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

import download_wan_models_006 as identity
from download_candidate_models_20261023 import OfficialRedirects, allowed_download_url, json_new


class TransferCancelled(Exception):
    pass


def check_stop(deadline, stop):
    if stop.is_set():
        raise TransferCancelled("Cooperative stop requested; owned partial preserved")
    if time.monotonic() >= deadline:
        raise TimeoutError("Transfer deadline reached; owned partial preserved")


def digest(path, deadline, stop):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            check_stop(deadline, stop)
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def verified(path, row, deadline, stop):
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == row["size"]
            and digest(path, deadline, stop) == row["sha256"])


def download_one(row, target, deadline, stop, opener=None):
    """Same disk/Range/publication contract as v001, with cooperative checks."""
    url = row["download_url"]
    if not allowed_download_url(url) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
        raise ValueError("Unreviewed official source or malformed SHA")
    identity.no_links(target)
    check_stop(deadline, stop)
    if target.exists():
        if not verified(target, row, deadline, stop):
            raise ValueError("Existing final differs; refusing overwrite")
        return {"status": "reused_verified", "size": row["size"], "sha256": row["sha256"]}
    partial = target.with_name(target.name + ".part")
    state = target.with_name(target.name + ".part.json")
    expected = {"size": row["size"], "sha256": row["sha256"]}
    identity.no_links(partial)
    identity.no_links(state)
    if partial.exists() and (not state.is_file() or identity.read(state) != expected):
        raise ValueError("Unowned or mismatched partial; preserved for inspection")
    if state.exists() and identity.read(state) != expected:
        raise ValueError("Partial identity differs")
    if not state.exists():
        json_new(state, expected)
    if not partial.exists():
        partial.touch(exist_ok=False)
    initial = partial.stat().st_size
    if initial > row["size"]:
        raise ValueError("Owned partial exceeds expected size")
    if initial == row["size"]:
        if not verified(partial, row, deadline, stop):
            raise ValueError("Completed partial SHA differs; preserved")
        check_stop(deadline, stop)
        if target.exists():
            raise ValueError("Final appeared concurrently; refusing replacement")
        os.replace(partial, target)
        state.unlink()
        return {"status": "promoted_verified_partial", **expected}
    opener = opener or urllib.request.build_opener(OfficialRedirects())
    for attempt in range(3):
        check_stop(deadline, stop)
        offset = partial.stat().st_size
        headers = {"User-Agent": "guoxin-immutable-checkpoint-downloader/2.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(request, timeout=min(30.0, max(1.0, deadline-time.monotonic()))) as response:
                check_stop(deadline, stop)
                if not allowed_download_url(response.geturl()):
                    raise ValueError("Final response host is not reviewed")
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != row["size"] or int(match[2]) >= row["size"]:
                        raise ValueError("Server Range differs from expected checkpoint")
                elif response.status == 200:
                    offset = 0  # Original v001 behavior when the server ignores Range.
                else:
                    raise ValueError("Unexpected HTTP download status")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > row["size"] - offset:
                    raise ValueError("Server payload exceeds expected size")
                with partial.open("r+b") as stream:
                    if not offset:
                        stream.truncate(0)
                    stream.seek(offset)
                    while True:
                        check_stop(deadline, stop)
                        block = response.read(min(4*1024*1024, row["size"]-stream.tell()+1))
                        check_stop(deadline, stop)
                        if not block:
                            break
                        if stream.tell()+len(block) > row["size"]:
                            raise ValueError("Downloaded bytes exceed expected size")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size != row["size"]:
                raise OSError("Incomplete response; owned partial preserved")
            if not verified(partial, row, deadline, stop):
                raise ValueError("Disk SHA failed; partial preserved")
            check_stop(deadline, stop)
            if target.exists():
                raise ValueError("Final appeared concurrently; refusing replacement")
            os.replace(partial, target)
            state.unlink()
            return {"status": "downloaded_verified", "resumed_from_bytes": initial, **expected}
        except (OSError, urllib.error.URLError, socket.timeout):
            if attempt == 2:
                raise
            check_stop(deadline, stop)
            stop.wait(min(1.0, max(0.0, deadline-time.monotonic())))
    raise RuntimeError("Unreachable retry state")


def run_workers(rows, destination, deadline, stop, workers, opener_factory=None):
    """Main-thread cancellation always precedes joining active workers."""
    if not 1 <= workers <= 3 or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Require 1..3 workers and unique file destinations")
    for row in rows:
        identity.check_relative(row["path"])
    results = {}
    interrupted = False

    def work(row):
        try:
            target = destination / row["path"]
            identity.no_links(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            result = download_one(row, target, deadline, stop,
                                  opener_factory(row) if opener_factory else None)
            return {**result, "path": row["path"], "source": row["download_url"]}
        except TransferCancelled as exc:
            return {"path": row["path"], "status": "cancelled_partial_preserved", "error": str(exc)}
        except Exception as exc:
            stop.set()
            return {"path": row["path"], "status": "failed_partial_preserved",
                    "error": repr(exc), "traceback": traceback.format_exc()}

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wan-owned-transfer")
    futures = {}
    try:
        futures = {executor.submit(work, row): row for row in rows}
        pending = set(futures)
        while pending:
            done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            done.update(future for future in pending if future.cancelled())
            for future in done:
                row = futures[future]
                result = ({"path": row["path"], "status": "cancelled_before_start"}
                          if future.cancelled() else future.result())
                results[row["path"]] = result
                print(json.dumps({"event": "file_result", **result}), flush=True)
            pending.difference_update(done)
            if stop.is_set():
                for future in pending:
                    future.cancel()
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
    except BaseException:
        stop.set()
        raise
    finally:
        if stop.is_set():
            for future in futures:
                future.cancel()
        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            if previous is not None:
                signal.signal(signal.SIGINT, previous)
        for future, row in futures.items():
            if row["path"] not in results:
                results[row["path"]] = ({"path": row["path"], "status": "cancelled_before_start"}
                                        if future.cancelled() else future.result())
    return list(results.values()), interrupted


def remaining_bytes(rows, destination):
    needed = 0
    for row in rows:
        target = destination / row["path"]
        identity.no_links(target)
        if target.exists():
            if not target.is_file() or target.stat().st_size != row["size"]:
                raise ValueError("Existing final size/type differs")
            continue
        partial = target.with_name(target.name + ".part")
        state = target.with_name(target.name + ".part.json")
        identity.no_links(partial)
        identity.no_links(state)
        present = 0
        if partial.exists():
            if not state.is_file() or identity.read(state) != {"size": row["size"], "sha256": row["sha256"]}:
                raise ValueError("Unowned or mismatched partial; no transfer started")
            present = partial.stat().st_size
            if present > row["size"]:
                raise ValueError("Partial exceeds expected size")
        needed += row["size"] - present
    return needed


def apply(rows, max_seconds, workers):
    if sys.platform != "linux" or not 1 <= workers <= 3 or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Require SCNet Linux, 1..3 workers, and unique paths")
    destination = identity.BASE / identity.NAMESPACE
    identity.no_links(destination)
    identity.BASE.mkdir(parents=True, exist_ok=True)
    lock_path = identity.BASE / "wan006-transfer.lock"
    identity.no_links(lock_path)
    token = uuid.uuid4().hex
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"run_id": token, "pid": os.getpid(), "version": "parallel_v002", "max_workers": workers}, stream)
    report = {"schema": "wan-checkpoint-receipt006-v2", "run_id": token, "status": "failed",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "destination": str(destination),
        "model": identity.MODEL, "revision": identity.REVISION, "max_workers": workers,
        "files": [], "GPU_imported": False, "all_workers_exited_before_lock_release": False}
    stop = threading.Event()
    try:
        needed = remaining_bytes(rows, destination)
        report["remaining_network_bytes_before_start"] = needed
        if shutil.disk_usage(identity.BASE).free < needed + 1024**3:
            raise ValueError("Insufficient free storage and reserve")
        # Start unfinished large files first; every existing final is still rehashed.
        ordered = sorted(rows, key=lambda row: ((destination/row["path"]).is_file(), -row["size"]))
        report["files"], interrupted = run_workers(ordered, destination, time.monotonic()+max_seconds, stop, workers)
        report["all_workers_exited_before_lock_release"] = True
        passed = {"downloaded_verified", "reused_verified", "promoted_verified_partial"}
        if interrupted:
            report["status"] = "interrupted_all_workers_stopped_partials_preserved"
            return 130
        if len(report["files"]) == 36 and all(row["status"] in passed for row in report["files"]):
            report["status"] = "completed_all_36_files_verified"
            return 0
        report["status"] = "failed_all_workers_stopped_partials_preserved"
        return 2
    except BaseException as exc:
        stop.set()
        report.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        json_new(identity.BASE / ("wan006-transfer-v002-"+token+".json"), report)
        if lock_path.is_file() and identity.read(lock_path).get("run_id") == token:
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    rows = identity.validate(identity.read(identity.SOURCES), identity.read(identity.CHECKPOINT))
    if args.action == "plan":
        print(json.dumps({"files": 36, "total_bytes": sum(row["size"] for row in rows),
            "destination": str(identity.BASE/identity.NAMESPACE), "workers": args.workers,
            "lock": str(identity.BASE/"wan006-transfer.lock"), "network_executed": False,
            "requires_old_transfer_exited_and_lock_absent": True, "GPU_imported": False}))
        return 0
    if not 60 <= args.max_seconds <= 21600:
        parser.error("Deadline must be in 60..21600 seconds")
    return apply(rows, args.max_seconds, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
