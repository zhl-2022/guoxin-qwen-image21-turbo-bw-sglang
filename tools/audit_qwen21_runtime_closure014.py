"""Audit separately exported runtime metadata, exact owner exits and HCU state."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
from audit_qwen21_full014_actual import read_archive, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--full-raw", type=Path, required=True)
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contents, payloads = read_archive(args.archive)
    assert payloads == 3
    full = json.loads((args.full_raw / "qwen21-complete-pretrained014-results/complete-pretrained014.json").read_text())
    launch = json.loads((args.full_raw / "e/qwen21-full014-B001-launch.json").read_text())
    cleanup = json.loads((args.full_raw / "e/qwen21-full014-B001-cleanup.json").read_text())
    runtime = json.loads(contents["e/qwen21-full014-B001-runtime.json"])
    driver = json.loads(contents["e/qwen21-full014-B001-driver-cleanup.json"])
    assert runtime["platform_notebook_id"] == launch["platform_id"] == cleanup["platform_notebook_id"] == driver["platform_notebook_id"]
    assert cleanup["launch"] == launch and cleanup["same_owned_process_present"] is False
    assert driver["driver_pid"] == full["pid"] and driver["driver_process_absent"] is True
    command = launch["command"]
    assert command.count(runtime["executable"]) == 1
    assert command[command.index(runtime["executable"]) + 1] == "-B"
    assert runtime["sys_version"].startswith("3.10.12 ")
    assert runtime["packages"]["sglang"]["version"] == full["native_before"]["version"]
    assert runtime["packages"]["torch"]["version"] == "2.10.0+das.opt1.dtk2604.20260325.g6b060a"
    for package, version in (("diffusers", "0.37.0"), ("transformers", "5.5.0"),
            ("Pillow", "12.1.1"), ("triton", "3.6.0+gitc73250c4.staging")):
        assert runtime["packages"][package]["version"] == version
    hcu = contents["e/qwen21-full014-B001-hy-smi.txt"]
    assert hcu == (args.full_raw / "e/qwen21-full014-B001-hy-smi.txt").read_bytes()
    values = [line.split() for line in hcu.decode().splitlines() if line.strip().startswith("0 ")]
    assert len(values) == 1 and values[0][5] == "0%" and values[0][6] == "0.0%"
    result = {"schema": "qwen21-full014-runtime-closure-independent-v1", "passed": True,
        "archive_bytes": args.archive.stat().st_size, "archive_sha256": sha(args.archive.read_bytes()),
        "CRC_manifest_payload_SHA_verified": True, "payloads": payloads,
        "platform_notebook_id": runtime["platform_notebook_id"], "runtime": runtime,
        "outer_timeout_pid": launch["pid"], "outer_start_ticks": launch["starttime"],
        "same_owned_outer_process_absent": True, "driver_pid": driver["driver_pid"], "driver_process_absent": True,
        "driver_absent_observed_utc": driver["UTC"], "HCU_VRAM_percent": 0, "HCU_utilization_percent": 0,
        "in_process_post_cleanup_allocated_bytes": full["post_cleanup_GPU_allocated"],
        "in_process_post_cleanup_reserved_bytes": full["post_cleanup_GPU_reserved"],
        "scope": "In-process cache allocation and later independently observed process/device cleanup are separate facts"}
    target = args.extract.absolute()
    assert not target.exists() and not any(p.is_symlink() for p in (target, *target.parents))
    target.mkdir(parents=True)
    for name, raw in contents.items():
        path = target.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream: stream.write(raw)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
