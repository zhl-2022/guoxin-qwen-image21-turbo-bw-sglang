"""Audit a bound DAS overlay and probe real components without model weights.

Run workers in separate processes: imports never synthesize framework modules.
CPU workers select the installed, real CpuPlatform implementation explicitly.
No service, GPU tensor, model download, or native file mutation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", SGLANG_FORCE_FUSED_OP_BACKEND="native")
sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def check_bundle(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = {row["path"] for row in manifest["files"]} | {"MANIFEST.json"}
    actual = set()
    for base, dirs, files in os.walk(root, onerror=lambda error: (_ for _ in ()).throw(error)):
        for item in dirs + files:
            path = Path(base) / item
            if path.is_symlink():
                raise ValueError("Bundle symlink refused: " + str(path))
        actual.update((Path(base) / name).relative_to(root).as_posix() for name in files)
    if actual != expected:
        raise ValueError("Bundle inventory mismatch: " + repr(sorted(actual ^ expected)))
    for row in manifest["files"]:
        path = root / row["path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Bundle bytes differ: " + row["path"])
    return {"files": len(manifest["files"]), "sha256": digest(root / "MANIFEST.json")}


def native_snapshot(root):
    lock = json.loads((root / "source-lock.json").read_text())
    dist = importlib.metadata.distribution("sglang")
    base = Path(dist.locate_file("")).resolve()
    mismatch = []
    for path, row in lock["native_wheel_python_files"].items():
        current = base / path
        if not current.is_file() or current.is_symlink() or digest(current) != row["sha256"]:
            mismatch.append(path)
    if mismatch:
        raise ValueError("Installed DAS source differs from fixed wheel: " + repr(mismatch[:20]))
    return {"version": dist.version, "root": str(base),
            "python_files_verified": len(lock["native_wheel_python_files"]),
            "wheel_sha256": lock["native_wheel_sha256"]}


def activate(root):
    overlay = root / "overlay"
    class BoundSourceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang" and not fullname.startswith("sglang."):
                return None
            local = overlay / Path(*fullname.split("."))
            if local.with_suffix(".py").is_file():
                return importlib.util.spec_from_file_location(fullname, local.with_suffix(".py"))
            if (local / "__init__.py").is_file():
                return importlib.util.spec_from_file_location(fullname, local / "__init__.py")
            # Existing native packages execute their original __init__.py.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                if spec.submodule_search_locations is not None and local.is_dir():
                    spec.submodule_search_locations = [str(local), *list(spec.submodule_search_locations)]
                return spec
            if local.is_dir():
                spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(local)]
                return spec
            return None
    # A standard source finder loads only real, hash-bound files. Native
    # package initializers remain real; no synthetic framework module exists.
    if any(key == "sglang" or key.startswith("sglang.") for key in sys.modules):
        raise RuntimeError("Activate before any framework import")
    sys.meta_path.insert(0, BoundSourceFinder())
    return {"finder": "BoundSourceFinder", "overlay": str(overlay)}


def setup_cpu():
    import torch
    torch.set_num_threads(2)
    torch.set_default_device("cpu")
    torch.manual_seed(42)
    # Explicitly select the actual DAS CPU platform before importing models.
    platforms = importlib.import_module("sglang.multimodal_gen.runtime.platforms")
    from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
    platforms._current_platform = CpuPlatform()
    return torch


def tensor_record(tensor):
    import torch
    assert tensor.device.type == "cpu", tensor.device
    assert torch.isfinite(tensor).all(), "Non-finite output"
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "abs_max": float(tensor.abs().max()), "finite": True,
            "sha256": hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()}


def init_cpu_parallel(root):
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel)
    rendezvous = root / ("gloo-" + str(os.getpid()))
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                distributed_init_method=rendezvous.as_uri(), timeout=30)
    initialize_model_parallel(sequence_parallel_degree=1, tensor_parallel_degree=1,
                              backend="gloo")


def init_native_srt_args():
    # Real SRT ServerArgs, in its documented no-serving initialization mode.
    # Qwen weights/model configs are separately bound by the MM execution path.
    from sglang.srt.server_args import (
        ServerArgs as SRTServerArgs, set_global_server_args_for_scheduler,
        get_global_server_args)
    args = SRTServerArgs(model_path="None", device="cpu", rl_on_policy_target=None)
    set_global_server_args_for_scheduler(args)
    assert get_global_server_args() is args and args.rl_on_policy_target is None
    return {"class": type(args).__module__ + "." + type(args).__name__,
            "purpose": "Auxiliary native RoPE configuration; no SRT model server",
            "model_path": args.model_path, "device": args.device,
            "rl_on_policy_target": args.rl_on_policy_target}


def run_component(name, root, output):
    torch = setup_cpu()
    if name == "scheduler_processor":
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import Qwen3VLProcessor
        assets = root / "model-assets"
        index = json.loads((assets / "model_index.json").read_text())
        expected = [1.0, .978453, .95418, .926626, .89508, .845148, .704534, .414568]
        assert index["sample_sigmas"] == expected
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(assets / "scheduler", local_files_only=True)
        scheduler.set_timesteps(sigmas=expected, device="cpu", mu=0.5)
        processor = Qwen3VLProcessor.from_pretrained(assets / "processor", local_files_only=True)
        from PIL import Image
        batch = processor(text=["Describe this object: <|vision_start|><|image_pad|><|vision_end|>"],
                          images=[Image.new("RGB", (32, 32), color=(128, 128, 128))],
                          return_tensors="pt", padding=True)
        assert len(scheduler.timesteps) == 8
        return {"components": {key: index[key] for key in ("processor", "text_encoder", "transformer", "vae", "scheduler")},
                "sigmas_input": expected, "sigmas_actual": scheduler.sigmas.tolist(),
                "timesteps": scheduler.timesteps.tolist(), "processor_class": type(processor).__name__,
                "batch_shapes": {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}}
    if name == "config":
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
        config = QwenImage21PipelineConfig()
        sigmas = json.loads((root / "model-assets/model_index.json").read_text())["sample_sigmas"]
        config.sample_sigmas = sigmas
        actual = config.prepare_sigmas(None, None)
        assert actual == sigmas, (actual, sigmas)
        return {"pipeline_config": type(config).__name__, "sigmas": actual}
    if name == "vae":
        from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEArchConfig, QwenImage21VAEConfig
        from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
        arch = QwenImage21VAEArchConfig(base_dim=4, decoder_base_dim=4, z_dim=4,
            dim_mult=(1, 2, 4, 4, 4), num_res_blocks=1,
            temperal_downsample=(False, False, False, False), in_channels=4, out_channels=4,
            latents_mean=(0.0,) * 4, latents_std=(1.0,) * 4)
        config = QwenImage21VAEConfig(arch_config=arch)
        config.use_parallel_decode = config.use_parallel_tiling = False
        model = AutoencoderKLQwenImage21(config).float().cpu().eval()
        with torch.inference_mode():
            latent = model.encode(torch.randn(1, 4, 1, 32, 64)).mode()
            decoded = model.decode(latent)
        assert latent.shape == (1, 4, 1, 2, 4) and decoded.shape == (1, 4, 1, 32, 64)
        return {"parameters": sum(p.numel() for p in model.parameters()),
                "latent": tensor_record(latent), "decoded": tensor_record(decoded)}
    init_cpu_parallel(output)
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    args = ServerArgs(model_path="Qwen/Qwen-Image-2.1-Turbo", num_gpus=1,
                      attention_backend="torch_sdpa", enable_torch_compile=False)
    set_global_server_args(args)
    init_native_srt_args()
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    if name == "dit":
        from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21ArchConfig, QwenImage21DitConfig
        from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import QwenImage21Transformer2DModel, build_layout
        config = QwenImage21DitConfig(arch_config=QwenImage21ArchConfig(in_channels=4, out_channels=4,
            num_layers=2, num_attention_heads=4, attention_head_dim=32, context_in_dim=16,
            mlp_ratio=2, axes_dims_rope=(8, 12, 12)))
        model = QwenImage21Transformer2DModel(config, {}).float().cpu().eval()
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=.02)
        layout = build_layout([False] * 3, [(1, 4, 4)], (8, 12, 12), "cpu")
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            result = model(hidden_states=torch.randn(1, 16, 4), encoder_hidden_states=torch.randn(1, 3, 16),
                timestep=torch.tensor([500.]), layouts=[layout])
        assert result.shape == (1, 16, 4)
        return {"parameters": sum(p.numel() for p in model.parameters()), "output": tensor_record(result)}
    if name == "encoder":
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as HFQwen3VLConfig
        from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLForConditionalGeneration
        arch = HFQwen3VLConfig(text_config=dict(hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, pad_token_id=0, rope_scaling=dict(rope_type="default", mrope_section=[2, 3, 3])),
            vision_config=dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=4,
                patch_size=2, temporal_patch_size=1, in_channels=3, num_position_embeddings=16,
                spatial_merge_size=2, out_hidden_size=64, deepstack_visual_indexes=[]),
            image_token_id=8, video_token_id=9, vision_start_token_id=7, vision_end_token_id=6)
        arch._fsdp_shard_conditions = []; arch.stacked_params_mapping = []
        model = Qwen3VLForConditionalGeneration(SimpleNamespace(arch_config=arch, quant_config=None)).float().cpu().eval()
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=.02)
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            result = model(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                           use_cache=False, output_hidden_states=True, logits_to_keep=1)
        assert result.hidden_states[-1].shape == (1, 4, 64)
        return {"parameters": sum(p.numel() for p in model.parameters()),
                "conditioning": tensor_record(result.hidden_states[-1]), "logits": tensor_record(result.logits)}
    raise ValueError(name)


def worker(name, root, output):
    record = {"component": name, "passed": False, "GPU_tensor_compute": False,
              "weights_loaded": False, "official_accuracy_verified": False}
    started = time.monotonic()
    try:
        record["native_before"] = native_snapshot(root)
        record["activated_packages"] = activate(root)
        record["result"] = run_component(name, root, output)
        record["passed"] = True
    except BaseException:
        record["traceback"] = traceback.format_exc()
    finally:
        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            record["cuda_context_initialized"] = torch.cuda.is_initialized()
            record["torch_version"] = torch.__version__
            record["torch_hip"] = torch.version.hip
        except BaseException:
            record["cleanup_traceback"] = traceback.format_exc()
        try:
            record["native_after"] = native_snapshot(root)
            record["native_unchanged"] = record["native_after"] == record.get("native_before")
        except BaseException:
            record["native_after_traceback"] = traceback.format_exc()
        record["module_sources"] = {key: {"path": value.__file__, "sha256": digest(value.__file__)}
            for key, value in list(sys.modules.items()) if key.startswith("sglang.")
            and getattr(value, "__file__", None) and Path(value.__file__).is_file()
            and value.__file__.endswith(".py")}
        record["elapsed_seconds"] = time.monotonic() - started
        write_new(output / (name + ".json"), record)
    return 0 if record["passed"] and record.get("native_unchanged") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--worker", choices=["config", "scheduler_processor", "vae", "dit", "encoder"])
    parser.add_argument("--components", nargs="+", choices=["scheduler_processor", "config", "vae", "dit", "encoder"],
                        default=["scheduler_processor", "config", "vae", "dit", "encoder"])
    args = parser.parse_args()
    root = args.bundle.resolve(strict=True)
    audit = check_bundle(root)
    if args.audit_only:
        print(json.dumps({"bundle": audit, "installed": False, "executed": False})); return 0
    if args.output is None:
        raise ValueError("A new output directory outside bundle is required")
    output = args.output.absolute()
    if output == root or root in output.parents or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Unsafe output directory")
    if args.worker:
        return worker(args.worker, root, output)
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Dynamic probes require native Linux CPython3.10 DAS environment")
    output.mkdir(parents=True, exist_ok=False)
    native = native_snapshot(root)
    workers = []
    for name in dict.fromkeys(args.components):
        with (output / (name + ".log")).open("x", encoding="utf-8") as stream:
            try:
                result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--bundle", str(root),
                    "--output", str(output), "--worker", name], stdout=stream, stderr=subprocess.STDOUT, timeout=150)
                workers.append({"name": name, "returncode": result.returncode})
            except subprocess.TimeoutExpired:
                workers.append({"name": name, "timeout_seconds": 150})
    receipt = {"bundle": audit, "native": native, "workers": workers,
               "passed": all(item.get("returncode") == 0 for item in workers),
               "components_selected": list(dict.fromkeys(args.components)),
               "all_five_components_selected": len(set(args.components)) == 5,
               "model_weights_loaded": False, "GPU_tensor_compute": False, "full_service_ready": False}
    write_new(output / "probe-summary.json", receipt)
    print(json.dumps(receipt, indent=2)); return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
