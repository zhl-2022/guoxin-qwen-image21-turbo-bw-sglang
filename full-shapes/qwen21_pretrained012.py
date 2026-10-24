"""Exact full architectures, bijective checkpoint plan and native streaming load."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path


def build_full_models(assets):
    import torch
    from types import SimpleNamespace
    from transformers import Qwen3VLConfig as HFConfig
    from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
    from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21ArchConfig, QwenImage21DitConfig
    from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEArchConfig, QwenImage21VAEConfig
    from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLForConditionalGeneration
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import QwenImage21Transformer2DModel
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

    def config_values(path, arch_type):
        raw = json.loads(path.read_text())
        accepted = {field.name for field in dataclasses.fields(arch_type)}
        unknown = [name for name in raw if name not in accepted and not name.startswith("_")]
        if unknown:
            raise ValueError("Unconsumed exact architecture fields: " + repr(unknown))
        return {name: value for name, value in raw.items() if name in accepted}

    hf_config = HFConfig.from_pretrained(assets / "text_encoder", local_files_only=True)
    hf_config._fsdp_shard_conditions = []; hf_config.stacked_params_mapping = []
    dit_config = QwenImage21DitConfig(arch_config=QwenImage21ArchConfig(**config_values(
        assets / "transformer/config.json", QwenImage21ArchConfig)))
    vae_config = QwenImage21VAEConfig(arch_config=QwenImage21VAEArchConfig(**config_values(
        assets / "vae/config.json", QwenImage21VAEArchConfig)))
    vae_config.use_tiling = vae_config.use_temporal_tiling = False
    vae_config.use_parallel_decode = vae_config.use_parallel_tiling = False
    with set_default_torch_dtype(torch.bfloat16), torch.device("meta"):
        models = {
            "text_encoder": Qwen3VLForConditionalGeneration(SimpleNamespace(arch_config=hf_config, quant_config=None)),
            "transformer": QwenImage21Transformer2DModel(dit_config, json.loads((assets / "transformer/config.json").read_text())),
            "vae": AutoencoderKLQwenImage21(vae_config),
        }
    return models


def map_name(component, name):
    # Exact concrete Qwen3VL.load_weights conversion. These published headers
    # contain no rotary_emb.inv_freq entry and no stacked/merge mapping.
    if component == "text_encoder" and "visual." in name:
        return name.replace(".attn.qkv.", ".attn.qkv_proj.")
    return name


def header_plan(directory, models):
    receipt = json.loads((directory / "header-receipt012.json").read_text())
    expected_lock = "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"
    assert receipt["source_lock_sha256"] == expected_lock and len(receipt["files"]) == 4
    mapped = {component: {} for component in models}
    source_count = {component: 0 for component in models}
    for row in receipt["files"]:
        raw = (directory / row["saved_header"]).read_bytes()
        assert len(raw) == row["actual_header_bytes"] and hashlib.sha256(raw).hexdigest() == row["actual_header_sha256"]
        component = row["component"]
        for name, value in json.loads(raw).items():
            if name == "__metadata__":
                continue
            target = map_name(component, name)
            assert "rotary_emb.inv_freq" not in name
            assert target not in mapped[component], "Duplicate mapped parameter: " + target
            mapped[component][target] = {"source_name": name, "shape": value["shape"],
                "dtype": value["dtype"], "checkpoint_file": row["path"]}
            source_count[component] += 1
    results = {}
    for component, model in models.items():
        expected = model.state_dict()
        names = set(expected)
        actual = set(mapped[component])
        missing, unexpected = sorted(names - actual), sorted(actual - names)
        shape_mismatch = [{"name": name, "model": list(expected[name].shape), "checkpoint": mapped[component][name]["shape"]}
            for name in sorted(names & actual) if list(expected[name].shape) != mapped[component][name]["shape"]]
        assert not missing and not unexpected and not shape_mismatch, {
            "component": component, "missing": missing, "unexpected": unexpected, "shape_mismatch": shape_mismatch}
        assert all(value["dtype"] == "BF16" for value in mapped[component].values())
        results[component] = {"source_keys": source_count[component], "target_keys": len(expected),
            "bijection_and_shapes_verified": True, "mapped_names_sha256": hashlib.sha256(json.dumps(mapped[component], sort_keys=True).encode()).hexdigest(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "all_parameters_meta": all(parameter.is_meta for parameter in model.parameters()),
            "meta_buffers": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in model.named_buffers() if value.is_meta],
            "custom_weight_loader_parameters": sum(hasattr(parameter, "weight_loader") for parameter in model.parameters()),
            "missing_key_synthesis_permitted": False, "merge_or_duplicate_keys_permitted": False}
    return results


def readonly_weight_iterator(root, rows):
    from safetensors import safe_open
    seen = set()
    for row in rows:
        path = root / row["path"]
        assert path.is_file() and not path.is_symlink()
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                assert name not in seen, "Duplicate checkpoint key: " + name
                seen.add(name)
                yield name, handle.get_tensor(name)


def load_component_native(model, component, checkpoint_root, rows):
    """Preserve native custom parameter loaders after strict header-plan approval.

    Caller must first validate whole-file disk SHAs and header_plan for this exact
    model. Native streaming loader receives a one-to-one mapping, so legacy merge
    buffering and missing-parameter synthesis cannot be reached for this plan.
    """
    import torch
    from sglang.multimodal_gen.runtime.loader.fsdp_load import load_model_from_streaming_state_dict
    device = torch.device("cuda", 0)
    mapping = lambda name: (map_name(component, name), None, None)
    result = load_model_from_streaming_state_dict(model,
        readonly_weight_iterator(checkpoint_root, rows), device, torch.bfloat16,
        strict=True, cpu_offload=False, param_names_mapping=mapping)
    assert not result.missing_keys and not result.unexpected_keys
    assert not any(parameter.is_meta for parameter in model.parameters())
    # Recompute only the exact two known nonpersistent positional caches using
    # their genuine original methods/classes. No parameter or unknown buffer is
    # synthesized. Dtypes follow the already-constructed native meta buffer.
    rebuilt = []
    for module_name, module in model.named_modules():
        for name, value in list(module._buffers.items()):
            if value is None:
                continue
            if value.is_meta:
                if name == "cos_sin_cache" and type(module).__module__.startswith("sglang.srt.layers.rotary_embedding."):
                    actual = module._compute_cos_sin_cache()
                elif name == "inv_freq" and type(module).__name__ == "Qwen3VLVisionRotaryEmbedding":
                    actual = type(module)(module.dim).inv_freq
                else:
                    raise ValueError("Unsupported unmaterialized buffer: " + module_name + "." + name)
                module._buffers[name] = actual.to(device=device, dtype=value.dtype)
                rebuilt.append(module_name + "." + name)
            elif value.device != device:
                module._buffers[name] = value.to(device=device)
    for module in model.modules():
        method = getattr(getattr(module, "quant_method", None), "process_weights_after_loading", None)
        if method is not None:
            method(module)
    post = getattr(model, "post_load_weights", None)
    if post is not None:
        post()
    assert not any(value.is_meta for value in (*model.parameters(), *model.buffers()))
    model.eval().requires_grad_(False)
    return {"missing": [], "unexpected": [], "native_streaming_custom_loaders_preserved": True,
        "positional_buffers_rebuilt_by_original_code": rebuilt}
