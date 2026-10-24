"""Strictly load the real published Qwen21 VAE and run small actual CPU forwards."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

BUNDLE_MANIFEST_SHA256 = "76befcfb98ae1b272e09abad9e1db1bf319c662be1c7c36ff2208ff1af9fed3b"
SOURCE_LOCK_SHA256 = "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"
CHECKPOINT = Path("/root/private_data/guoxin-candidates/models/Qwen-Image-2.1-Turbo-fc13ba1986f")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def write_new(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().with_name("download-sources009.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise ValueError("Actual checkpoint probe requires native Linux CPython3.10 DAS interpreter")
    if not 60 <= args.timeout_seconds <= 900:
        raise ValueError("Probe timeout must be60..900 seconds")
    bundle = args.bundle.resolve(strict=True)
    checkpoint = args.checkpoint.absolute()
    output = args.output.absolute()
    if checkpoint != CHECKPOINT or any(path.is_symlink() for path in (checkpoint, *checkpoint.parents)):
        raise ValueError("Require the independently owned exact Qwen checkpoint namespace")
    if output.exists() or any(path.is_symlink() for path in (output, *output.parents)) or bundle in output.parents or checkpoint in output.parents:
        raise ValueError("Fresh independent output directory required")
    if digest(bundle / "MANIFEST.json") != BUNDLE_MANIFEST_SHA256 or digest(args.lock) != SOURCE_LOCK_SHA256:
        raise ValueError("Frozen009 bundle/source lock identity differs")
    output.mkdir(parents=True)
    started = time.monotonic()
    record = {"schema": "qwen21-actual-vae-checkpoint010-v1", "passed": False,
        "checkpoint_directory": str(checkpoint), "model_revision": "fc13ba1986fcb905854dc8de2f52eee395f49b5b",
        "GPU_tensor_compute": False, "full_model_weights_loaded": False, "official_accuracy_verified": False,
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256, "source_lock_sha256": SOURCE_LOCK_SHA256,
        "verified_files": [], "CPU_compute_dtype": "float32", "strict_checkpoint_load": False}
    # A Unix alarm requests a bounded exit; an external controller can also
    # enforce a process timeout if a native call stops returning to Python.
    import signal
    def deadline(signum, frame):
        raise TimeoutError("VAE actual-checkpoint probe deadline")
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    core = None
    try:
        for row in json.loads(args.lock.read_text())["files"]:
            if row["component"] not in ("metadata", "vae"):
                continue
            path = checkpoint / row["path"]
            if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
                raise ValueError("Actual phase bytes differ: " + row["path"])
            record["verified_files"].append({"path": row["path"], "bytes": row["size"], "sha256": row["sha256"]})
        assert len(record["verified_files"]) == 14
        spec = importlib.util.spec_from_file_location("qwen21_core010_source", bundle / "probe_qwen21_core007.py")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        record["bundle"] = core.check_bundle(bundle)
        record["native_before"] = core.native_snapshot(bundle)
        record["activated_packages"] = core.activate(bundle)
        torch = core.setup_cpu()
        from safetensors import safe_open
        from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEArchConfig, QwenImage21VAEConfig
        from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
        source_config = json.loads((checkpoint / "vae/config.json").read_text())
        assert source_config["_class_name"] == "AutoencoderKLQwenImage21"
        accepted = {field.name for field in dataclasses.fields(QwenImage21VAEArchConfig)}
        unknown = sorted(key for key in source_config if key not in accepted and not key.startswith("_"))
        assert not unknown, unknown
        config = QwenImage21VAEConfig(arch_config=QwenImage21VAEArchConfig(**{
            key: value for key, value in source_config.items() if key in accepted}))
        config.use_tiling = config.use_parallel_tiling = config.use_parallel_decode = False
        config.use_temporal_tiling = False
        with torch.device("meta"):
            model = AutoencoderKLQwenImage21(config)
        with safe_open(checkpoint / "vae/diffusion_pytorch_model.safetensors", framework="pt", device="cpu") as handle:
            state = {name: handle.get_tensor(name) for name in handle.keys()}
            names = sorted(state)
            shapes = {name: list(tensor.shape) for name, tensor in state.items()}
            checkpoint_dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
            expected = model.state_dict()
            assert set(expected) == set(state), {"missing": sorted(set(expected) - set(state)),
                "unexpected": sorted(set(state) - set(expected))}
            assert all(expected[name].shape == state[name].shape for name in expected)
            del expected
            result = model.load_state_dict(state, strict=True, assign=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert not any(parameter.is_meta for parameter in model.parameters())
        record["strict_checkpoint_load"] = True
        record["checkpoint_keys"] = len(names)
        record["checkpoint_keys_sha256"] = hashlib.sha256(json.dumps(names).encode()).hexdigest()
        record["checkpoint_shapes_sha256"] = hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()
        record["checkpoint_dtypes"] = checkpoint_dtypes
        record["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        del state
        model.float().cpu().eval()
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        record["loaded_seconds"] = time.monotonic() - started
        with torch.inference_mode():
            image = torch.linspace(-1, 1, 4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 1, 32, 32)
            latent = model.encode(image).mode()
            decoded = model.decode(latent)
        assert latent.shape == (1, 64, 1, 2, 2) and decoded.shape == (1, 4, 1, 32, 32)
        record["latent"] = core.tensor_record(latent)
        record["decoded"] = core.tensor_record(decoded)
        record["passed"] = True
    except BaseException:
        record["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        try:
            if core is not None and "native_before" in record:
                record["native_after"] = core.native_snapshot(bundle)
                record["native_unchanged"] = record["native_after"] == record["native_before"]
            if "torch" in sys.modules:
                record["cuda_context_initialized"] = sys.modules["torch"].cuda.is_initialized()
        except BaseException:
            record["cleanup_traceback"] = traceback.format_exc()
        record["elapsed_seconds"] = time.monotonic() - started
        write_new(output / "vae-checkpoint010.json", record)
    print(json.dumps(record, indent=2))
    return 0 if record["passed"] and record.get("native_unchanged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
