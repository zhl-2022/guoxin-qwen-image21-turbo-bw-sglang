"""Independently audit actual phase/complete-model originals without modifying them."""
from __future__ import annotations
import argparse
import hashlib
import json
import io
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--driver", type=Path, action="append", default=[])
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "evidence/session007/qwen21-pipeline-prep009/download-sources009.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    lock = {row["path"]: row for row in json.loads(args.lock.read_text())["files"]}
    bundles = {sha((path / "MANIFEST.json").read_bytes()): json.loads((path / "source-lock.json").read_text()) for path in args.bundle}
    result = {"archive_sha256": sha(args.archive.read_bytes()), "archive_bytes": args.archive.stat().st_size,
        "CRC_verified": False, "manifest_verified": False, "evidence": []}
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        result["CRC_verified"] = True
        names = archive.namelist()
        assert len(names) == len({name.casefold() for name in names})
        for info in archive.infolist():
            rel = PurePosixPath(info.filename)
            assert not rel.is_absolute() and ".." not in rel.parts and "\\" not in info.filename and ":" not in info.filename
            assert rel.as_posix() == info.filename and "\x00" not in info.filename
            assert not stat.S_ISLNK(info.external_attr >> 16) and not info.is_dir() and info.file_size < 64 * 1024 * 1024
        manifest_names = set(names) & {"MANIFEST.json", "export-manifest.json"}
        assert len(manifest_names) == 1
        manifest_name = manifest_names.pop()
        manifest = json.loads(archive.read(manifest_name))["files"]
        if isinstance(manifest, list):
            rows = manifest
            manifest = {row["path"]: row for row in rows}
            assert len(manifest) == len(rows)
        assert isinstance(manifest, dict) and set(names) == set(manifest) | {manifest_name}
        for name, row in manifest.items():
            raw = archive.read(name)
            assert len(raw) == row.get("bytes", row.get("size")) and sha(raw) == row["sha256"]
        result.update(manifest_verified=True, manifest_payloads=len(manifest))
        for name in names:
            if not name.endswith(".json") or name == "MANIFEST.json":
                continue
            record = json.loads(archive.read(name))
            if PurePosixPath(name).name not in ("vae-checkpoint010.json", "full-shapes012.json", "complete-pretrained013.json", "complete-pretrained014.json"):
                continue
            source_lock = bundles[record["bundle"]["sha256"]]
            before, after = record["native_before"], record["native_after"]
            assert before == after and record["native_unchanged"]
            assert before["wheel_sha256"] == source_lock["native_wheel_sha256"]
            assert before["python_files_verified"] == len(source_lock["native_wheel_python_files"])
            entry = {"path": name, "passed": record["passed"], "native_sources_unchanged": True,
                "native_source_count": before["python_files_verified"], "bundle_bound": True,
                "cuda_context_initialized": record.get("cuda_context_initialized"),
                "GPU_tensor_compute": record["GPU_tensor_compute"], "elapsed_seconds": record["elapsed_seconds"]}
            if record["passed"]:
                if PurePosixPath(name).name == "full-shapes012.json":
                    assert record["checkpoint_tensors_loaded"] is False
                    expected_counts = {"text_encoder": 750, "transformer": 297, "vae": 238}
                    expected_parameters = {"text_encoder": 8767123696, "transformer": 7115124736, "vae": 337740404}
                    for key, count in expected_counts.items():
                        row = record["components"][key]
                        assert row["source_keys"] == row["target_keys"] == count
                        assert row["bijection_and_shapes_verified"] and row["all_parameters_meta"]
                        assert row["parameters"] == expected_parameters[key]
                    entry["components"] = record["components"]
                else:
                    expected_count = 14 if PurePosixPath(name).name == "vae-checkpoint010.json" else 17
                    files = {row["path"]: row for row in record["verified_files"]}
                    assert len(files) == len(record["verified_files"]) == expected_count
                    for path, row in files.items():
                        assert row["sha256"] == lock[path]["sha256"] and row.get("bytes", row.get("size")) == lock[path]["size"]
                    if expected_count == 14:
                        assert record["strict_checkpoint_load"] and record["GPU_tensor_compute"] is False
                        assert record["parameters"] == 337740404 and record["checkpoint_keys"] == 238
                        assert record["latent"]["finite"] and record["decoded"]["finite"]
                        assert record["latent"]["device"] == record["decoded"]["device"] == "cpu"
                        entry.update(parameters=record["parameters"], checkpoint_keys=record["checkpoint_keys"], latent=record["latent"], decoded=record["decoded"])
                    else:
                        assert record["all_weights_loaded"] and record["GPU_tensor_compute"]
                        assert record["samples"] and record["official_accuracy_verified"] is False
                        for driver in args.driver:
                            if all((driver / filename).is_file() and sha((driver / filename).read_bytes()) == value
                                   for filename, value in record["driver_sources"].items()):
                                entry["driver_sources_bound"] = True
                                break
                        if args.driver:
                            assert entry.get("driver_sources_bound") is True
                        if PurePosixPath(name).name == "complete-pretrained014.json":
                            for component, loaded in record["loads"].items():
                                assert not loaded["missing"] and not loaded["unexpected"]
                                assert len(loaded["actual_parameter_value_checks"]) >= 2
                                assert all(check["exact_equal"] and check["actual_sha256"] == check["source_sha256"]
                                    for check in loaded["actual_parameter_value_checks"])
                        for sample in record["samples"]:
                            path = str(PurePosixPath(name).parent / sample["png"])
                            if path.startswith("./"): path = path[2:]
                            raw = archive.read(path)
                            assert len(raw) == sample["bytes"] and sha(raw) == sample["sha256"]
                            assert raw[:8] == b"\x89PNG\r\n\x1a\n"
                            assert int.from_bytes(raw[16:20], "big") == sample["width"]
                            assert int.from_bytes(raw[20:24], "big") == sample["height"] and raw[25] == 6
                            from PIL import Image
                            with Image.open(io.BytesIO(raw)) as picture:
                                picture.verify()
                            with Image.open(io.BytesIO(raw)) as picture:
                                picture.load()
                                assert picture.mode == "RGBA" and picture.size == (sample["width"], sample["height"])
                            assert len(sample["chain"]["steps"]) == 8 and sample["chain"]["pre_final_norm_verified"]
                            assert sample["chain"]["decoded"]["finite"]
                        entry["samples"] = [{key: sample[key] for key in ("height", "width", "bytes", "sha256", "elapsed_seconds_including_validation")} for sample in record["samples"]]
            else:
                entry["traceback"] = record.get("traceback")
            result["evidence"].append(entry)
        assert result["evidence"], "No recognized actual-stage receipt"
        if args.extract:
            target = args.extract.absolute()
            assert not target.exists() and not any(path.is_symlink() for path in (target, *target.parents))
            target.mkdir(parents=True)
            for name in names:
                path = target.joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(archive.read(name))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
