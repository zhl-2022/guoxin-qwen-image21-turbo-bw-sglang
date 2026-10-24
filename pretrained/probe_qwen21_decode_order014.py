"""Regress the actual BF16 reciprocal/divide/add decoding contract on real Torch."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_default_device("cpu"); torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("actual_stage014", args.stage)
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
    with zipfile.ZipFile(args.archive) as archive:
        prefix = archive.namelist()[0] + "python/"
        raw = archive.read(prefix + "sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py")
    functions = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
        and node.name in ("scale_and_shift_latents", "_to_unit_range")]
    assert len(functions) == 2
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "exact-upstream-decoding-functions", "exec"), namespace)
    config = json.loads((args.assets / "vae/config.json").read_text())
    arch = SimpleNamespace(**config)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        latents = torch.linspace(-3, 3, 64 * 2 * 4, dtype=dtype).reshape(1, 64, 1, 2, 4)
        mean = latents.new_tensor(arch.latents_mean).view(1, 64, 1, 1, 1)
        std = latents.new_tensor(arch.latents_std).view(1, 64, 1, 1, 1)
        # Only a math/config fixture, never a synthetic framework/model module.
        args_fixture = SimpleNamespace(pipeline_config=SimpleNamespace(
            get_decode_scale_and_shift=lambda device, dt, vae: (std.reciprocal(), mean)))
        exact = namespace["scale_and_shift_latents"](latents, args_fixture, None)
        actual = stage.denormalize_latents(latents, arch)
        old_multiply = latents * std + mean
        assert torch.equal(actual, exact)
        differences = int(torch.count_nonzero(exact != old_multiply))
        if dtype == torch.bfloat16:
            assert differences > 0, "BF16 regression case must distinguish old operation order"
        pixels = torch.linspace(-1, 1, 1024, dtype=dtype)
        exact_pixels = namespace["_to_unit_range"](pixels)
        actual_pixels = (pixels / 2 + .5).clamp(0, 1)
        assert torch.equal(actual_pixels, exact_pixels)
        cases.append({"dtype": str(dtype), "exact_original_scale_shift_equal": True,
            "exact_original_pixel_normalization_equal": True,
            "old_multiply_diff_elements": differences,
            "old_multiply_max_abs": float((exact.float() - old_multiply.float()).abs().max())})
    result = {"passed": True, "cases": cases, "upstream_decode_source_sha256": hashlib.sha256(raw).hexdigest(),
        "stage_sha256": hashlib.sha256(args.stage.read_bytes()).hexdigest(), "Torch": torch.__version__,
        "GPU_tensor_compute": False, "native_DAS": False, "full_checkpoint_loaded": False,
        "scope": "Unmodified original arithmetic functions executed on real Torch CPU; no native component runtime claim"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
