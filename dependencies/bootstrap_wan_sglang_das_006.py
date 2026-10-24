#!/usr/bin/env python3
"""Install a reviewed offline wheel set into a new Linux Python3.10 venv only.

No dependency resolution, system package install, model load, or network access.
The root coordinator separately authorizes and starts GPU runtime probing.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv
import zipfile

ALLOWED = {"sglang", "sglang-kernel", "diffusers", "addict", "imageio", "imageio-ffmpeg",
           "orjson", "apache-tvm-ffi", "moviepy", "proglog", "python-dotenv"}
PROTECTED = ("torch", "triton", "torchvision", "torchaudio", "transformers", "vllm", "vllm-hcu")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def audit_wheels(bundle: Path) -> list[dict]:
    manifest = json.loads((bundle / "wheels-manifest.json").read_text(encoding="utf-8"))
    rows = manifest["wheels"]
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Duplicate wheel filename")
    names = set()
    for row in rows:
        if Path(row["filename"]).name != row["filename"] or not row["filename"].endswith(".whl"):
            raise ValueError("Unsafe wheel filename")
        wheel = bundle / "wheels" / row["filename"]
        if wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size != row["bytes"] or sha(wheel) != row["sha256"]:
            raise ValueError("Wheel identity mismatch: " + row["filename"])
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None:
                raise ValueError("Wheel CRC failed")
            entries = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise ValueError("Ambiguous wheel metadata")
            info = email.message_from_bytes(archive.read(entries[0]))
            name = canonical(info["Name"])
            if name != canonical(row["name"]) or name not in ALLOWED or name in names:
                raise ValueError("Unapproved or duplicate distribution")
            if info["Version"] != row["version"]:
                raise ValueError("Wheel version mismatch")
            names.add(name)
    if not {"sglang", "sglang-kernel"}.issubset(names):
        raise ValueError("Matching DAS framework/kernel pair required")
    return rows


def native_snapshot() -> dict:
    result = {}
    for name in PROTECTED:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            result[name] = None
            continue
        files = []
        for item in dist.files or []:
            if ".dist-info/" in str(item) and Path(item).name in {"METADATA", "RECORD", "WHEEL", "entry_points.txt", "direct_url.json"}:
                path = Path(dist.locate_file(item))
                if path.is_file():
                    files.append({"path": str(path), "sha256": sha(path)})
        result[name] = {"version": dist.version, "root": str(dist.locate_file("")), "metadata": sorted(files, key=lambda x: x["path"])}
    return result


def write_new(path: Path, record) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows = audit_wheels(args.bundle.resolve(strict=True))
    if args.audit_only:
        print(json.dumps({"offline_wheels_audited": len(rows), "installed": False}, indent=2))
        return 0
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("This bundle targets the existing Linux CPython3.10 DAS image")
    if args.target is None:
        raise ValueError("A fresh isolated --target is required")
    target = args.target.absolute()
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Symlink installation target or ancestor refused")
    if not str(target).startswith("/root/private_data/") or target.exists():
        raise ValueError("Use a new directory under /root/private_data; no overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    before = native_snapshot()
    write_new(target / "native-before.json", before)
    record = {"wheel_count": len(rows), "native_changed": None, "pip_returncode": None,
              "dependency_resolution_enabled": False, "network_enabled": False,
              "GPU_inference_executed": False, "venv": str(target / "venv")}
    try:
        # The native Ubuntu image has no ensurepip/python3.10-venv package.
        # Reuse its pip module through system-site-packages, while installation
        # paths are supplied by the new interpreter's distinct sys.prefix.
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(target / "venv")
        python = target / "venv/bin/python"
        prefix_check = subprocess.run([str(python), "-c",
            "import json,sys,sysconfig; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'purelib':sysconfig.get_paths()['purelib']}))"],
            capture_output=True, text=True, timeout=30, check=True)
        prefix = json.loads(prefix_check.stdout)
        expected_prefix = (target / "venv").resolve()
        if (Path(prefix["prefix"]).resolve() != expected_prefix
                or prefix["prefix"] == prefix["base_prefix"]
                or not Path(prefix["purelib"]).resolve().is_relative_to(expected_prefix)):
            raise RuntimeError("New interpreter does not install into the isolated venv")
        write_new(target / "venv-prefix-check.json", prefix)
        command = [str(python), "-m", "pip", "--disable-pip-version-check", "install",
                   "--no-index", "--no-deps", "--no-compile"]
        command.extend(str(args.bundle.resolve() / "wheels" / x["filename"]) for x in rows)
        with (target / "pip-install.log").open("x", encoding="utf-8") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300,
                                 env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        record["pip_returncode"] = run.returncode
    finally:
        after = native_snapshot()
        write_new(target / "native-after.json", after)
        record["native_changed"] = before != after
        write_new(target / "bootstrap-result.json", record)
    print(json.dumps(record, indent=2))
    if record["pip_returncode"] != 0 or record["native_changed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
