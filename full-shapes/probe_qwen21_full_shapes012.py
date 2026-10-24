"""Compare genuine full native meta models with actual published tensor headers."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--headers", type=Path, default=Path(__file__).resolve().with_name("headers"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == "linux" and sys.version_info[:2] == (3, 10)
    root = args.bundle.resolve(strict=True)
    assert hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest() == args.expected_bundle_sha256
    output = args.output.absolute()
    assert not output.exists() and root not in output.parents
    assert not any(path.is_symlink() for path in (output, *output.parents))
    output.mkdir(parents=True)
    result = {"passed": False, "GPU_tensor_compute": False, "checkpoint_tensors_loaded": False,
        "full_model_architectures": True, "source": "Actual immutable published HTTP206 tensor headers; not whole-file SHA proof"}
    start = time.monotonic()
    core = None
    def timeout(signum, frame):
        raise TimeoutError("Full native architecture metadata probe timeout")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(240)
    try:
        spec = importlib.util.spec_from_file_location("qwen_core012", root / "probe_qwen21_core007.py")
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        result["bundle"] = core.check_bundle(root)
        result["native_before"] = core.native_snapshot(root)
        result["activation"] = core.activate(root)
        torch = core.setup_cpu()
        core.init_cpu_parallel(output)
        from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
        set_global_server_args(ServerArgs(model_path="Qwen/Qwen-Image-2.1-Turbo", num_gpus=1,
            attention_backend="torch_sdpa", enable_torch_compile=False))
        result["srt_auxiliary_config"] = core.init_native_srt_args()
        from qwen21_pretrained012 import build_full_models, header_plan
        models = build_full_models(root / "model-assets")
        result["components"] = header_plan(args.headers, models)
        assert all(row["all_parameters_meta"] for row in result["components"].values())
        result["passed"] = True
    except BaseException:
        result["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        try:
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                result["cuda_context_initialized"] = torch.cuda.is_initialized()
            if core and "native_before" in result:
                result["native_after"] = core.native_snapshot(root)
                result["native_unchanged"] = result["native_before"] == result["native_after"]
        except BaseException:
            result["cleanup_traceback"] = traceback.format_exc()
        result["elapsed_seconds"] = time.monotonic() - start
        with (output / "full-shapes012.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] and result.get("native_unchanged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
