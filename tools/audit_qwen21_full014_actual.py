"""Independently bind full014 originals to source, checkpoint and image contracts.

This reads only the supplied archive and frozen local source. It never imports
SGLang, Torch, network clients or the model runner, and never executes a model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
LOCK_SHA = "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"
BUNDLE_SHA = "73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d"
MODEL = "Qwen/Qwen-Image-2.1-Turbo"
DESTINATION = "/root/private_data/guoxin-candidates/models/Qwen-Image-2.1-Turbo-fc13ba1986f"
SIGMAS = [1.0, .978453, .95418, .926626, .89508, .845148, .704534, .414568, 0.]
KEYS = {"text_encoder": 750, "transformer": 297, "vae": 238}
PARAMETERS = {"text_encoder": 8767123696, "transformer": 7115124736, "vae": 337740404}
DRIVERS = ("run_qwen21_pretrained014.py", "qwen21_pretrained014.py", "qwen21_t2i_stage014.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_archive(path):
    """Check CRC, exact inventory, every payload SHA and extraction-safe names."""
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        infos = archive.infolist()
        assert len(infos) == len({row.filename.casefold() for row in infos})
        assert sum(row.file_size for row in infos) < 256 * 1024**2
        for row in infos:
            rel = PurePosixPath(row.filename)
            assert rel.as_posix() == row.filename and not rel.is_absolute()
            assert not any(part in ("..", ".") for part in rel.parts)
            assert not any(token in row.filename for token in ("\\", ":", "\0"))
            assert not row.is_dir() and not stat.S_ISLNK(row.external_attr >> 16)
            assert row.file_size < 64 * 1024**2
        manifest_names = {row.filename for row in infos} & {"MANIFEST.json", "export-manifest.json"}
        assert len(manifest_names) == 1
        manifest_name = manifest_names.pop()
        manifest = json.loads(archive.read(manifest_name))["files"]
        if isinstance(manifest, list):
            rows = manifest
            manifest = {row["path"]: row for row in rows}
            assert len(manifest) == len(rows)
        assert isinstance(manifest, dict)
        assert {row.filename for row in infos} == set(manifest) | {manifest_name}
        contents = {row.filename: archive.read(row.filename) for row in infos}
        for name, expected in manifest.items():
            assert len(contents[name]) == expected.get("bytes", expected.get("size"))
            assert sha(contents[name]) == expected["sha256"]
    return contents, len(manifest)


def files_bound(rows, expected):
    actual = {row["path"]: row for row in rows}
    assert len(rows) == len(actual) == 17 and set(actual) == set(expected)
    for name, row in actual.items():
        assert row.get("size", row.get("bytes")) == expected[name]["size"]
        assert row["sha256"] == expected[name]["sha256"]
    return sum(row["size"] for row in expected.values())


def transfer_contract(record, expected):
    assert record["schema"] == "qwen21-turbo009-transfer-receipt-v1"
    assert record["model"] == MODEL and record["hf_revision"] == REVISION
    assert record["destination"] == DESTINATION
    assert record["phase"] == "all" and record["max_workers"] == 2
    assert record["mode"] in ("download", "verify")
    assert record["status"] == "completed_selected_phase_full_SHA_verified"
    assert record["full_checkpoint_verified"] is True
    assert record["all_workers_exited_before_lock_release"] is True
    assert record["GPU_imported"] is False and record["torch_in_sys_modules"] is False
    assert re.fullmatch("[0-9a-f]{32}", record["run_id"])
    total = files_bound(record["files"], expected)
    allowed = {"verified"} if record["mode"] == "verify" else {
        "downloaded_verified", "reused_verified", "promoted_verified_partial"}
    assert all(row["status"] in allowed for row in record["files"])
    assert dt.datetime.fromisoformat(record["started_utc"]) <= dt.datetime.fromisoformat(record["finished_utc"])
    return {"run_id": record["run_id"], "mode": record["mode"], "files": 17,
        "total_bytes": total, "finished_utc": record["finished_utc"],
        "all_workers_joined": True, "GPU_imported": False}


def gpu_tensor(row, shape):
    assert row["shape"] == shape and row["finite"] is True
    assert row["device"] in ("cuda", "cuda:0")
    assert re.fullmatch("[0-9a-f]{64}", row["sha256_float32"])


def complete_contract(record, name, contents, expected, source_lock, driver):
    assert record["schema"] == "qwen21-complete-pretrained014-v1"
    assert record["model_id"] == MODEL and record["model_revision"] == REVISION
    assert record["checkpoint"] == DESTINATION
    assert record["bundle_manifest_sha256"] == record["bundle"]["sha256"] == BUNDLE_SHA
    assert set(record["driver_sources"]) == set(DRIVERS)
    for filename in DRIVERS:
        assert record["driver_sources"][filename] == sha((driver / filename).read_bytes())
    assert record["native_before"] == record["native_after"] and record["native_unchanged"] is True
    assert record["native_before"]["wheel_sha256"] == source_lock["native_wheel_sha256"]
    assert record["native_before"]["python_files_verified"] == len(source_lock["native_wheel_python_files"]) == 1989
    result = {"passed": record["passed"], "driver_sources_bound": True,
        "native_1989_files_unchanged": True, "GPU_tensor_compute": record["GPU_tensor_compute"],
        "GPU_tensors_loaded": record.get("GPU_tensors_loaded"),
        "cuda_context_initialized": record.get("cuda_context_initialized"),
        "elapsed_seconds": record["elapsed_seconds"], "samples": []}
    # Preserve a genuine failure as a failed run. Model upload or a context does
    # not prove a completed GPU forward, and a failed image does not prove none.
    if not record["passed"]:
        assert record.get("traceback")
        result.update(traceback=record["traceback"],
            completed_checkpoint_files=len(record["verified_files"]),
            cleanup_traceback=record.get("cleanup_traceback"))
        return result
    assert record["all_weights_loaded"] is True and record["GPU_tensor_compute"] is True
    assert record["GPU_tensors_loaded"] is True and record["cuda_context_initialized"] is True
    assert not record.get("cleanup_traceback")
    assert record["official_accuracy_verified"] is False
    assert record["official_performance_verified"] is False and record["submitted"] is False
    result["checkpoint_full_SHA_bytes"] = files_bound(record["verified_files"], expected)
    assert record["platform"]["class"] == "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform"
    assert record["platform"]["hip"] and record["platform"]["device_name"]
    assert set(record["shape_plan"]) == set(record["loads"]) == set(KEYS)
    for component, plan in record["shape_plan"].items():
        assert plan["source_keys"] == plan["target_keys"] == KEYS[component]
        assert plan["parameters"] == PARAMETERS[component]
        assert plan["bijection_and_shapes_verified"] and plan["all_parameters_meta"]
        loaded = record["loads"][component]
        assert loaded["native_streaming_custom_loaders_preserved"] is True
        assert not loaded["missing"] and not loaded["unexpected"]
        checks = loaded["actual_parameter_value_checks"]
        assert 2 <= len(checks) <= 4 and len({row["parameter"] for row in checks}) == len(checks)
        for row in checks:
            assert row["exact_equal"] and row["actual_sha256"] == row["source_sha256"]
            assert row["bytes"] <= 32 * 1024**2 and row["dtype"] == "torch.bfloat16"
        if component != "vae":
            assert any(row["native_custom_weight_loader"] for row in checks)
    samples = record["samples"]
    sizes = [(row["width"], row["height"]) for row in samples]
    assert sizes in ([(256, 256)], [(256, 256), (512, 512)], [(256, 256), (1024, 1024)])
    for sample in samples:
        width, height = sample["width"], sample["height"]
        assert sample["seed"] == 42 and sample["prompt"] == "A red cube on a clean white background."
        chain = sample["chain"]
        assert chain["drop_system_tokens"] == 14 and chain["pre_final_norm_verified"] is True
        assert len(chain["prompt_ids"]) == 1 and len(chain["prompt_ids"][0]) == 31
        assert chain["prefix_cache_layers_populated"] == 32 and chain["condition_reference_images"] == 0
        gpu_tensor(chain["conditioning"], [1, 17, 4096])
        assert len(chain["sigmas"]) == 9
        assert all(abs(actual - fixed) < 6e-8 for actual, fixed in zip(chain["sigmas"], SIGMAS))
        assert len(chain["steps"]) == 8
        for index, step in enumerate(chain["steps"]):
            assert step["index"] == index and abs(step["timestep"] - chain["sigmas"][index] * 1000) < .0001
            gpu_tensor(step["latents"], [1, (height // 16) * (width // 16), 64])
        gpu_tensor(chain["decoded"], [1, 4, 1, height, width])
        png_name = str(PurePosixPath(name).parent / sample["png"])
        assert PurePosixPath(sample["png"]).name == sample["png"]
        png = contents[png_name]
        assert len(png) == sample["bytes"] and sha(png) == sample["sha256"]
        from PIL import Image
        with Image.open(io.BytesIO(png)) as image:
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            image.load()
            assert image.mode == "RGBA" and image.size == (width, height)
            result["samples"].append({"width": width, "height": height, "sha256": sha(png),
                "bytes": len(png), "channel_extrema": image.getextrema(),
                "GPU_eight_steps_finite": True, "pre_norm_and_32_prefix_caches": True,
                "elapsed_seconds_including_validation": sample["elapsed_seconds_including_validation"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--bundle", type=Path, default=root / "candidates/qwen21-turbo-das007-010")
    parser.add_argument("--driver", type=Path, default=root / "tasks/qwen21-turbo-pretrained014")
    parser.add_argument("--lock", type=Path, default=root / "evidence/session007/qwen21-pipeline-prep009/download-sources009.json")
    args = parser.parse_args()
    lock_raw = args.lock.read_bytes()
    assert sha(lock_raw) == LOCK_SHA
    expected = {row["path"]: row for row in json.loads(lock_raw)["files"]}
    assert sha((args.bundle / "MANIFEST.json").read_bytes()) == BUNDLE_SHA
    source_lock = json.loads((args.bundle / "source-lock.json").read_text())
    contents, payloads = read_archive(args.archive)
    result = {"schema": "qwen21-full014-independent-contract-audit-v1",
        "archive_sha256": sha(args.archive.read_bytes()), "archive_bytes": args.archive.stat().st_size,
        "CRC_manifest_and_payload_SHA_verified": True, "payloads": payloads,
        "checkpoint_source_lock_sha256": LOCK_SHA, "transfer_receipts": [], "full014": [],
        "official_accuracy_verified": False, "official_performance_verified": False, "submitted": False}
    for name, raw in contents.items():
        if not name.endswith(".json") or name == "MANIFEST.json":
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            continue
        if record.get("schema") == "qwen21-turbo009-transfer-receipt-v1" and record.get("phase") == "all":
            result["transfer_receipts"].append({"path": name, **transfer_contract(record, expected)})
        elif record.get("schema") == "qwen21-complete-pretrained014-v1":
            result["full014"].append({"path": name, **complete_contract(record, name, contents, expected, source_lock, args.driver)})
    assert result["transfer_receipts"] or result["full014"], "No actual whole-checkpoint/full014 evidence found"
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
