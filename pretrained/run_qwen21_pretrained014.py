"""Strict complete Qwen21 Turbo TP1 HCU generation, one load and multiple sizes."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

CHECKPOINT = Path("/root/private_data/guoxin-candidates/models/Qwen-Image-2.1-Turbo-fc13ba1986f")
LOCK_SHA256 = "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def event(name, **values):
    print(json.dumps({"event": name, **values}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--shape-receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().with_name("download-sources009.json"))
    parser.add_argument("--headers", type=Path, default=Path(__file__).resolve().with_name("headers"))
    parser.add_argument("--resolutions", choices=("256", "256,512", "256,1024"), default="256,512")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    assert sys.platform == "linux" and sys.version_info[:2] == (3, 10)
    assert 120 <= args.max_seconds <= 3600
    bundle = args.bundle.resolve(strict=True)
    output = args.output.absolute()
    assert not output.exists() and bundle not in output.parents and CHECKPOINT not in output.parents
    assert not any(path.is_symlink() for path in (output, *output.parents, CHECKPOINT, *CHECKPOINT.parents))
    assert digest(bundle / "MANIFEST.json") == args.expected_bundle_sha256
    assert digest(args.lock) == LOCK_SHA256
    prior = json.loads(args.shape_receipt.read_text())
    assert prior["passed"] and prior["native_unchanged"]
    assert prior["bundle"]["sha256"] == args.expected_bundle_sha256
    assert all(row["bijection_and_shapes_verified"] and row["all_parameters_meta"] for row in prior["components"].values())
    lock = json.loads(args.lock.read_text())
    assert lock["hf_revision"] == "fc13ba1986fcb905854dc8de2f52eee395f49b5b" and len(lock["files"]) == 17
    output.mkdir(parents=True)
    record = {"schema": "qwen21-complete-pretrained014-v1", "passed": False,
        "model_id": "Qwen/Qwen-Image-2.1-Turbo", "model_revision": lock["hf_revision"],
        "checkpoint": str(CHECKPOINT), "bundle_manifest_sha256": args.expected_bundle_sha256,
        "shape_receipt_sha256": digest(args.shape_receipt), "pid": os.getpid(),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "verified_files": [], "samples": [],
        "GPU_tensor_compute": False, "all_weights_loaded": False, "official_accuracy_verified": False,
        "official_performance_verified": False, "submitted": False,
        "profile": {"TP": 1, "SP": 1, "batch": 1, "sigmas": 8, "attention": "torch_sdpa",
            "compile": False, "graphs": False, "quantization": False, "offload": False,
            "conditioning_cache": False, "new_BitExactFusionGate_paths": False},
        "scope": "Real complete pretrained development images; no official quality/performance claim"}
    record["driver_sources"] = {name: digest(Path(__file__).resolve().with_name(name)) for name in
        ("run_qwen21_pretrained014.py", "qwen21_pretrained014.py", "qwen21_t2i_stage014.py")}
    start = time.monotonic()
    core = None
    models = {}
    decoded = None
    def stop(signum, frame):
        raise TimeoutError("Owned complete Qwen generation deadline or signal: " + str(signum))
    for number in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM): signal.signal(number, stop)
    signal.alarm(args.max_seconds)
    try:
        # Whole-file integrity is checked once per process before loading.
        for row in lock["files"]:
            path = CHECKPOINT / row["path"]
            assert path.is_file() and not any(item.is_symlink() for item in (path, *path.parents))
            assert path.stat().st_size == row["size"] and digest(path) == row["sha256"], row["path"]
            record["verified_files"].append({"path": row["path"], "size": row["size"], "sha256": row["sha256"]})
            event("actual_file_SHA_verified", path=row["path"], size=row["size"])
        spec = importlib.util.spec_from_file_location("qwen_core013", bundle / "probe_qwen21_core007.py")
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        record["bundle"] = core.check_bundle(bundle)
        record["native_before"] = core.native_snapshot(bundle)
        record["activation"] = core.activate(bundle)
        import torch
        torch.set_num_threads(2); torch.set_default_device("cpu"); torch.manual_seed(42)
        assert torch.version.hip is not None and torch.cuda.is_available(), "Genuine HCU/HIP required"
        torch.cuda.set_device(0)
        # Actual HIP is already checked; select the genuine installed ROCm
        # platform explicitly so optional vendor-management detection cannot
        # misclassify HCU as NVIDIA. No implementation is replaced.
        import sglang.multimodal_gen.runtime.platforms as platforms
        from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform
        platforms._current_platform = RocmPlatform()
        current_platform = platforms.current_platform
        assert current_platform.is_rocm() and not current_platform.is_cuda(), "Require actual native ROCm platform"
        record["platform"] = {"class": type(current_platform).__module__ + "." + type(current_platform).__name__,
            "torch": torch.__version__, "hip": torch.version.hip, "device_name": torch.cuda.get_device_name(0)}
        free, total = torch.cuda.mem_get_info(0)
        record["admission_GPU_memory"] = {"free": free, "total": total}
        weights_bytes = sum(row["size"] for row in lock["files"] if row["path"].endswith(".safetensors"))
        assert free >= weights_bytes + 8 * 1024**3, "Insufficient currently free device memory and reserve"
        from sglang.multimodal_gen.runtime.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
            distributed_init_method=(output / "hcu-tp1-rendezvous").as_uri(), timeout=60)
        initialize_model_parallel(sequence_parallel_degree=1, tensor_parallel_degree=1, backend="nccl")
        from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
        pipeline_config = QwenImage21PipelineConfig()
        pipeline_config.sample_sigmas = json.loads((CHECKPOINT / "model_index.json").read_text())["sample_sigmas"]
        set_global_server_args(ServerArgs(model_path=str(CHECKPOINT), num_gpus=1,
            pipeline_config=pipeline_config, disable_autocast=True,
            attention_backend="torch_sdpa", enable_torch_compile=False))
        from sglang.srt.server_args import ServerArgs as SRTServerArgs, set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(SRTServerArgs(model_path="None", device="cuda", rl_on_policy_target=None))
        from qwen21_pretrained014 import build_full_models, header_plan, load_component_native
        models = build_full_models(bundle / "model-assets")
        record["shape_plan"] = header_plan(args.headers, models)
        event("actual_full_meta_key_shape_plan_passed", components=record["shape_plan"])
        # Disable genuine optional optimization gates; the unmodified original
        # eager reduction/SDPA/residual paths remain selected on actual HIP.
        from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
        disabled = []
        for module_name in ("sglang.multimodal_gen.runtime.models.dits.qwen_image21",
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21"):
            for name, value in vars(sys.modules[module_name]).items():
                if isinstance(value, BitExactFusionGate):
                    value.disable(); disabled.append(module_name + "." + name)
        record["disabled_optional_gates"] = disabled
        record["loads"] = {}
        for component in ("text_encoder", "transformer", "vae"):
            rows = [row for row in lock["files"] if row["component"] == component]
            event("native_streaming_load_started", component=component)
            record["loads"][component] = load_component_native(models[component], component, CHECKPOINT, rows)
            event("native_streaming_load_complete", component=component)
        record["all_weights_loaded"] = True
        record["GPU_tensors_loaded"] = True
        record["load_and_SHA_seconds"] = time.monotonic() - start
        from transformers import Qwen3VLProcessor
        from diffusers import FlowMatchEulerDiscreteScheduler
        from qwen21_t2i_stage014 import run_t2i
        from PIL import Image
        processor = Qwen3VLProcessor.from_pretrained(CHECKPOINT / "processor", local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CHECKPOINT / "scheduler", local_files_only=True)
        sigmas = json.loads((CHECKPOINT / "model_index.json").read_text())["sample_sigmas"]
        def on_stage(name, **values):
            if str(values.get("device", "")).startswith("cuda"):
                record["GPU_tensor_compute"] = True
            event(name, **values)
        for size in map(int, args.resolutions.split(",")):
            sample_start = time.monotonic()
            event("complete_pretrained_sample_started", height=size, width=size)
            record["GPU_compute_attempted"] = True
            # Until a genuine component forward completes, a partial exception
            # does not prove absence of GPU work.
            if record["GPU_tensor_compute"] is False:
                record["GPU_tensor_compute"] = None
            decoded, chain = run_t2i(text_encoder=models["text_encoder"], dit=models["transformer"], vae=models["vae"],
                processor=processor, scheduler=scheduler, sigmas=sigmas,
                prompt="A red cube on a clean white background.", height=size, width=size, seed=42, progress=on_stage)
            assert decoded.device.type == "cuda"
            rgba = ((decoded[0, :, 0] / 2 + 0.5).clamp(0, 1)).float().permute(1, 2, 0).cpu().numpy()
            path = output / ("qwen21-pretrained014-" + str(size) + ".png")
            Image.fromarray((rgba * 255).round().astype("uint8"), mode="RGBA").save(path)
            torch.cuda.synchronize()
            sample = {"height": size, "width": size, "seed": 42, "prompt": "A red cube on a clean white background.",
                "png": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "chain": chain,
                "elapsed_seconds_including_validation": time.monotonic() - sample_start,
                "GPU_memory_peak_allocated": torch.cuda.max_memory_allocated(0),
                "GPU_memory_peak_reserved": torch.cuda.max_memory_reserved(0)}
            record["samples"].append(sample); record["GPU_tensor_compute"] = True
            decoded = None
            event("complete_pretrained_sample_saved", height=size, width=size, sha256=sample["sha256"])
        record["passed"] = True
    except BaseException:
        record["traceback"] = traceback.format_exc()
        event("complete_pretrained_run_failed", last_line=record["traceback"].strip().splitlines()[-1])
    finally:
        signal.alarm(0)
        models.clear(); decoded = None
        try:
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    record["post_cleanup_GPU_allocated"] = torch.cuda.memory_allocated(0)
                    record["post_cleanup_GPU_reserved"] = torch.cuda.memory_reserved(0)
                record["cuda_context_initialized"] = torch.cuda.is_initialized()
            if core and "native_before" in record:
                record["native_after"] = core.native_snapshot(bundle)
                record["native_unchanged"] = record["native_before"] == record["native_after"]
        except BaseException:
            record["cleanup_traceback"] = traceback.format_exc()
        record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["elapsed_seconds"] = time.monotonic() - start
        with (output / "complete-pretrained014.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2); stream.write("\n")
    event("complete_pretrained_run_finished", passed=record["passed"], output=str(output))
    return 0 if record["passed"] and record.get("native_unchanged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
