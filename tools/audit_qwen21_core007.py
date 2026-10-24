"""Check exact source binding and mathematical invariants of the serial port."""
from __future__ import annotations
import ast
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidates/qwen21-turbo-das007-006"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def method(raw, cls, name):
    module = ast.parse(raw.decode("utf-8"))
    node = next(item for item in module.body if isinstance(item, ast.ClassDef) and item.name == cls)
    return next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name)


def body(node):
    return ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/session007/qwen21-core006-source-invariant-audit.json")
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    checks = []
    def check(name, condition, detail=None):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True, "detail": detail})
    lock = json.loads((candidate / "source-lock.json").read_text())
    archive_path = ROOT / "dist/sglang-qwen21-efc984-source007.zip"
    wheel_path = ROOT / "evidence/session006/wan-preflight/hygon/wheels" / lock["native_wheel"]
    check("exact_upstream_archive", sha(archive_path.read_bytes()) == lock["upstream_source_archive_sha256"])
    check("native_DAS_wheel_unchanged", sha(wheel_path.read_bytes()) == lock["native_wheel_sha256"])
    with zipfile.ZipFile(archive_path) as archive, zipfile.ZipFile(wheel_path) as wheel:
        prefix = archive.namelist()[0] + "python/"
        check("source_and_native_CRC", archive.testzip() is None and wheel.testzip() is None)
        for row in lock["files"]:
            relative = row["path"].removeprefix("overlay/")
            upstream = archive.read(prefix + relative)
            current = (candidate / row["path"]).read_bytes()
            assert sha(upstream) == row["upstream_sha256"]
            assert sha(current) == row["sha256"] and len(current) == row["bytes"]
            assert hashlib.sha1(f"blob {len(upstream)}\0".encode() + upstream).hexdigest() == row["upstream_git_blob_sha1"]
            if row["path"].endswith(".py"):
                ast.parse(current.decode("utf-8"), feature_version=(3, 10))
        python_modules = sum(row["path"].endswith(".py") for row in lock["files"])
        check("all_real_source_files_bound_and_Python310_parsed", python_modules > 0,
              {"source_files": len(lock["files"]), "Python_modules": python_modules})
        native_norm = method(wheel.read("sglang/srt/layers/layernorm.py"), "RMSNorm", "forward_native")
        current_norm = method(archive.read(prefix + "sglang/srt/layers/layernorm.py"), "RMSNorm", "forward_native")
        check("RMSNorm_native_reduction_AST_identical", body(native_norm) == body(current_norm),
              "Main adds an unused quant_linear argument; forward_native statements are identical.")
        vision_path = "sglang/multimodal_gen/runtime/models/encoders/qwen_vl_vision.py"
        check("vision_Torch_SDPA_fallback_AST_identical",
              body(method(archive.read(prefix + vision_path), "QwenVLVisionAttention", "_packed_attention")) ==
              body(method((candidate / "overlay" / vision_path).read_bytes(), "QwenVLVisionAttention", "_packed_attention")))
        dit_path = "sglang/multimodal_gen/runtime/models/dits/qwen_image21.py"
        dit = (candidate / "overlay" / dit_path).read_bytes()
        for name in ("forward", "prepare_modulation"):
            check("DiT_" + name + "_AST_identical", body(method(dit, "QwenImage21Transformer2DModel", name)) ==
                  body(method(archive.read(prefix + dit_path), "QwenImage21Transformer2DModel", name)))
        encoder_path = "sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py"
        for cls, name in (("Qwen3VLTextModel", "forward"), ("Qwen3VLForConditionalGeneration", "forward"),
                          ("Qwen3VLForConditionalGeneration", "load_weights")):
            check(cls + "_" + name + "_AST_identical",
                  body(method((candidate / "overlay" / encoder_path).read_bytes(), cls, name)) ==
                  body(method(archive.read(prefix + encoder_path), cls, name)))
        vae_path = "sglang/multimodal_gen/runtime/models/vaes/autoencoder_kl_qwenimage21.py"
        check("VAE_encode_AST_identical",
              body(method((candidate / "overlay" / vae_path).read_bytes(), "AutoencoderKLQwenImage21", "_encode")) ==
              body(method(archive.read(prefix + vae_path), "AutoencoderKLQwenImage21", "_encode")))
    assets = candidate / "model-assets"
    index = json.loads((assets / "model_index.json").read_text())
    check("exact_five_Qwen21_components", {key: index[key][1] for key in ("processor", "text_encoder", "transformer", "vae", "scheduler")} == {
        "processor": "Qwen3VLProcessor", "text_encoder": "Qwen3VLForConditionalGeneration",
        "transformer": "QwenImage21Transformer2DModel", "vae": "AutoencoderKLQwenImage21",
        "scheduler": "FlowMatchEulerDiscreteScheduler"})
    check("exact_Turbo_eight_sigmas", index["sample_sigmas"] == [1.0, .978453, .95418, .926626, .89508, .845148, .704534, .414568])
    check("processor_LFS_bytes_hash", sha((assets / "processor/tokenizer.json").read_bytes()) == "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506")
    check("no_checkpoint_tensor", not list(candidate.rglob("*.safetensors")))
    result = {"passed": True, "checks": checks, "checks_passed": len(checks),
              "native_dynamic_import_verified": False, "GPU_executed": False,
              "full_model_weights_loaded": False, "official_accuracy_verified": False}
    path = args.output
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
