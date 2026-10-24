"""Close the actual CPU/full-architecture gate for a conditional full014 launch."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    core_audit = json.loads((root / "evidence/session007/qwen21-core010-independent-audit.json").read_text())
    stage_audit = json.loads((root / "evidence/session007/qwen21-stage010-012-independent-audit.json").read_text())
    module_audit = json.loads((root / "evidence/session007/qwen21-core010-loaded-source-bindings.json").read_text())
    assert core_audit["all_workers_passed"] and len(core_audit["workers"]) == 2
    assert stage_audit["manifest_verified"] and stage_audit["manifest_payloads"] == 25
    assert module_audit["passed"] and module_audit["distinct_framework_modules_bound"] == 699
    tiny_path = args.raw / "qwen21-tiny-pipeline011-results/tiny-pipeline011.json"
    tiny = json.loads(tiny_path.read_text())
    manifest_sha = "73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d"
    assert tiny["passed"] and tiny["bundle"]["sha256"] == manifest_sha
    assert tiny["native_before"] == tiny["native_after"] and tiny["native_unchanged"]
    assert tiny["native_before"]["python_files_verified"] == 1989
    assert not tiny["full_checkpoint_loaded"] and tiny["GPU_tensor_compute"] is False
    chain = tiny["chain"]
    assert chain["pre_final_norm_verified"] and chain["drop_system_tokens"] == 14
    assert chain["conditioning"]["shape"] == [1, 17, 64] and chain["conditioning"]["finite"]
    expected = [1., .978453, .95418, .926626, .89508, .845148, .704534, .414568, 0.]
    assert len(chain["sigmas"]) == 9 and all(abs(left - right) < 6e-8 for left, right in zip(chain["sigmas"], expected))
    assert len(chain["steps"]) == 8 and chain["prefix_cache_layers_populated"] == 2
    for index, step in enumerate(chain["steps"]):
        assert step["index"] == index and abs(step["timestep"] - 1000 * chain["sigmas"][index]) < .0001
        assert step["latents"]["finite"] and step["latents"]["device"] == "cpu"
    assert chain["decoded"]["shape"] == [1, 4, 1, 32, 32] and chain["decoded"]["finite"]
    png = tiny_path.with_name("tiny-random-chain011.png")
    assert sha(png) == tiny["PNG_sha256"]
    with Image.open(png) as picture: picture.verify()
    with Image.open(png) as picture:
        picture.load(); assert picture.size == (32, 32) and picture.mode == "RGBA"
    shape = json.loads((args.raw / "qwen21-full-shapes012-results/full-shapes012.json").read_text())
    vae = json.loads((args.raw / "qwen21-vae-checkpoint010-results/vae-checkpoint010.json").read_text())
    assert shape["passed"] and vae["passed"] and vae["strict_checkpoint_load"]
    expected_parameters = {"text_encoder": 8767123696, "transformer": 7115124736, "vae": 337740404}
    assert {key: value["parameters"] for key, value in shape["components"].items()} == expected_parameters
    assert len(shape["components"]["text_encoder"]["meta_buffers"]) == 2
    assert shape["components"]["text_encoder"]["custom_weight_loader_parameters"] == 232
    assert shape["components"]["transformer"]["custom_weight_loader_parameters"] == 224
    assert shape["components"]["vae"]["custom_weight_loader_parameters"] == 0
    decode = json.loads((root / "evidence/session007/qwen21-decode-order014-real-CPU-regression-b.json").read_text())
    package = root / "tasks/qwen21-turbo-pretrained014"
    assert decode["passed"] and decode["stage_sha256"] == sha(package / "qwen21_t2i_stage014.py")
    launch = json.loads((args.raw / "launch/qwen-all-transfer013-launch.json").read_text())
    result = {"schema": "qwen21-full014-readiness-gate-v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "actual_source_archive_sha256": stage_audit["archive_sha256"],
        "all_CPU_and_full_architecture_prerequisites_passed": True,
        "core010_selected_DiT_encoder_native_forwards_passed": True,
        "fullmeta012_published_key_shape_bijections_passed": True,
        "fullmeta012_parameters": expected_parameters,
        "genuine_native_custom_loaders": {"text_encoder": 232, "transformer": 224, "vae": 0},
        "actual_published_VAE_checkpoint_CPU_passed": True, "actual_VAE_parameters": vae["parameters"],
        "actual_metadata_plus_VAE14_files_full_SHA_bound": len(vae["verified_files"]) == 14,
        "tiny011_eight_sigma_five_component_chain_passed": True, "tiny011_pre_final_norm_verified": True,
        "tiny011_random_PNG_fully_decoded": True, "tiny011_PNG_sha256": tiny["PNG_sha256"],
        "014_exact_decode_order_original_function_regression_passed": True,
        "014_zip_sha256": sha(root / "dist/session007-qwen21-pretrained014.zip"),
        "014_helper_sha256": sha(root / "scripts/unpack_qwen21_pretrained014.py"),
        "native_files_per_process_verified_unchanged": 1989,
        "actual_all17_download_launch_UTC": dt.datetime.fromtimestamp(launch["epoch"], dt.timezone.utc).isoformat(),
        "actual_all17_download_pid": launch["pid"], "actual_all17_download_start_ticks": launch["start_ticks"],
        "all17_checkpoint_full_SHA_verified": False,
        "full014_GPU_launch_condition": "Launch after actual all17 full-SHA verification receipt succeeds and the owned transfer process has exited; retain native HCU admission checks",
        "full014_GPU_has_executed": False, "official_accuracy_verified": False, "official_performance_verified": False,
        "scope": "Actual native CPU prerequisites passed; full pretrained GPU generation remains conditional on actual complete checkpoint receipt"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
