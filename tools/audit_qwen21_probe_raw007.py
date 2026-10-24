"""Audit the CRC/manifest and actual per-worker evidence of a Qwen21 probe."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    source_lock = json.loads((bundle / "source-lock.json").read_text())
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None, "CRC failure"
        names = archive.namelist()
        assert len(names) == len(set(name.casefold() for name in names)), "Duplicate paths"
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
            assert ":" not in info.filename and not stat.S_ISLNK(info.external_attr >> 16)
            assert not info.is_dir() and info.file_size < 16 * 1024 * 1024
        manifest = json.loads(archive.read("MANIFEST.json"))["files"]
        assert set(manifest) | {"MANIFEST.json"} == set(names)
        for name, row in manifest.items():
            raw = archive.read(name)
            assert len(raw) == row["bytes"] and sha(raw) == row["sha256"], name
        summaries = [name for name in names if PurePosixPath(name).name == "probe-summary.json"]
        assert len(summaries) == 1, "Unique probe summary required"
        parent = str(PurePosixPath(summaries[0]).parent)
        prefix = "" if parent == "." else parent + "/"
        summary = json.loads(archive.read(summaries[0]))
        assert summary["bundle"]["sha256"] == sha((bundle / "MANIFEST.json").read_bytes())
        workers = []
        for item in summary["workers"]:
            name = item["name"]
            record = json.loads(archive.read(prefix + name + ".json"))
            before, after = record["native_before"], record["native_after"]
            assert before == after and record["native_unchanged"] is True
            assert before["wheel_sha256"] == source_lock["native_wheel_sha256"]
            assert before["python_files_verified"] == len(source_lock["native_wheel_python_files"])
            traceback = record.get("traceback", "")
            workers.append({"name": name, "returncode": item.get("returncode"),
                            "passed": record["passed"], "native_sources_verified": before["python_files_verified"],
                            "native_unchanged": True, "cuda_context_initialized": record.get("cuda_context_initialized"),
                            "elapsed_seconds": record["elapsed_seconds"], "result": record.get("result"),
                            "error_last_line": traceback.strip().splitlines()[-1] if traceback else None,
                            "traceback_sha256": sha(traceback.encode()) if traceback else None})
        if args.extract is not None:
            target = args.extract.absolute()
            assert not target.exists() and not any(path.is_symlink() for path in (target, *target.parents))
            target.mkdir(parents=True)
            for name in names:
                path = target.joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(archive.read(name))
    result = {"archive": str(args.archive.resolve()), "archive_bytes": args.archive.stat().st_size,
              "archive_sha256": sha(args.archive.read_bytes()), "CRC_verified": True,
              "manifest_payloads_verified": len(manifest), "bundle_manifest_bound": True,
              "worker_path_prefix": prefix,
              "native_wheel_sha256": source_lock["native_wheel_sha256"], "workers": workers,
              "all_workers_passed": all(row["passed"] for row in workers),
              "all_native_sources_unchanged": True,
              "failure_layer": "probe source finder before component loading" if all(
                  row["error_last_line"] == "AttributeError: '_NamespacePath' object has no attribute 'insert'" for row in workers)
                  else "see individual raw workers",
              "model_weights_loaded": summary["model_weights_loaded"],
              "GPU_tensor_compute": summary["GPU_tensor_compute"], "full_service_ready": summary["full_service_ready"]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
