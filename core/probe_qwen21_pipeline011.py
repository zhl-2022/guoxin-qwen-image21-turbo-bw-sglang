"""Exercise all five genuine Qwen21 component types as a tiny CPU eight-step chain."""
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
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise ValueError("Native Linux CPython3.10 required")
    root = args.bundle.resolve(strict=True)
    assert hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest() == args.expected_bundle_sha256
    output = args.output.absolute()
    assert not output.exists() and root not in output.parents
    assert not any(path.is_symlink() for path in (output, *output.parents))
    output.mkdir(parents=True)
    record = {"passed": False, "scope": "Five real component types; tiny random architectures, not pretrained image generation",
        "full_checkpoint_loaded": False, "GPU_tensor_compute": False, "official_accuracy_verified": False}
    start = time.monotonic()
    core = None
    def timeout(signum, frame):
        raise TimeoutError("Tiny five-component pipeline deadline")
    signal.signal(signal.SIGALRM, timeout)
    assert 60 <= args.timeout_seconds <= 900
    signal.alarm(args.timeout_seconds)
    try:
        spec = importlib.util.spec_from_file_location("qwen21_core011_source", root / "probe_qwen21_core007.py")
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        record["bundle"] = core.check_bundle(root)
        record["native_before"] = core.native_snapshot(root)
        record["activation"] = core.activate(root)
        torch = core.setup_cpu()
        core.init_cpu_parallel(output)
        from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
        set_global_server_args(ServerArgs(model_path="Qwen/Qwen-Image-2.1-Turbo", num_gpus=1,
            attention_backend="torch_sdpa", enable_torch_compile=False))
        record["srt_auxiliary_config"] = core.init_native_srt_args()
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import Qwen3VLProcessor, Qwen3VLConfig as HFConfig
        from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21ArchConfig, QwenImage21DitConfig
        from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEArchConfig, QwenImage21VAEConfig
        from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLForConditionalGeneration
        from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import QwenImage21Transformer2DModel
        from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
        from qwen21_t2i_stage011 import run_t2i
        assets = root / "model-assets"
        processor = Qwen3VLProcessor.from_pretrained(assets / "processor", local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(assets / "scheduler", local_files_only=True)
        vocabulary = max(processor.tokenizer.get_vocab().values()) + 1
        arch = HFConfig(text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=vocabulary, pad_token_id=0,
            rope_scaling=dict(rope_type="default", mrope_section=[2, 3, 3])),
            vision_config=dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=4, patch_size=2,
                temporal_patch_size=1, in_channels=3, num_position_embeddings=16, spatial_merge_size=2,
                out_hidden_size=64, deepstack_visual_indexes=[]),
            image_token_id=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            video_token_id=processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            vision_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_token_id=processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        arch._fsdp_shard_conditions = []; arch.stacked_params_mapping = []
        encoder = Qwen3VLForConditionalGeneration(SimpleNamespace(arch_config=arch, quant_config=None)).float().cpu().eval()
        dit = QwenImage21Transformer2DModel(QwenImage21DitConfig(arch_config=QwenImage21ArchConfig(in_channels=4,
            out_channels=4, num_layers=2, num_attention_heads=4, attention_head_dim=32, context_in_dim=64,
            mlp_ratio=2, axes_dims_rope=(8, 12, 12))), {}).float().cpu().eval()
        config = QwenImage21VAEConfig(arch_config=QwenImage21VAEArchConfig(base_dim=4, decoder_base_dim=4, z_dim=4,
            dim_mult=(1, 2, 4, 4, 4), num_res_blocks=1, temperal_downsample=(False,) * 4,
            in_channels=4, out_channels=4, latents_mean=(0.,) * 4, latents_std=(1.,) * 4))
        config.use_tiling = config.use_parallel_decode = config.use_parallel_tiling = False
        vae = AutoencoderKLQwenImage21(config).float().cpu().eval()
        for model in (encoder, dit, vae):
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=.02)
        record["parameters"] = {name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in (("text_encoder", encoder), ("dit", dit), ("vae", vae))}
        decoded, chain = run_t2i(text_encoder=encoder, dit=dit, vae=vae, processor=processor, scheduler=scheduler,
            sigmas=json.loads((assets / "model_index.json").read_text())["sample_sigmas"],
            prompt="A red cube on a clean white background.", height=32, width=32, seed=42)
        from PIL import Image
        rgba = ((decoded[0, :, 0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((rgba * 255).round().astype("uint8"), mode="RGBA").save(output / "tiny-random-chain011.png")
        record.update(passed=True, chain=chain, PNG_sha256=hashlib.sha256((output / "tiny-random-chain011.png").read_bytes()).hexdigest())
    except BaseException:
        record["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        try:
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                record["cuda_context_initialized"] = torch.cuda.is_initialized()
            if core and "native_before" in record:
                record["native_after"] = core.native_snapshot(root)
                record["native_unchanged"] = record["native_before"] == record["native_after"]
        except BaseException:
            record["cleanup_traceback"] = traceback.format_exc()
        record["elapsed_seconds"] = time.monotonic() - start
        with (output / "tiny-pipeline011.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2); stream.write("\n")
    print(json.dumps(record, indent=2))
    return 0 if record["passed"] and record.get("native_unchanged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
