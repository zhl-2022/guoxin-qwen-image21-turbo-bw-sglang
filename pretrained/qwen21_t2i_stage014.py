# SPDX-License-Identifier: Apache-2.0
# Prompt/layout math adapted from the fixed Qwen Team SGLang Qwen-Image2.1 stage:
# efc9849520216b31b8a4a4322399e47e8dbde067.
"""Minimal single-request real-model T2I orchestration for a strict TP1 profile."""
from __future__ import annotations
import hashlib
import time

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_TEMPLATE = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
SIGMAS = [1.0, .978453, .95418, .926626, .89508, .845148, .704534, .414568]


def collapse_image_slots(hidden, input_ids, image_token_id):
    image_mask = input_ids == image_token_id
    keep = ~image_mask
    keep[0] = True
    keep[1:] |= image_mask[1:] & ~image_mask[:-1]
    return hidden[keep], image_mask[keep]


def prompt_inputs(processor, prompt):
    """The original no-reference-image branch of upstream _encode_prompt."""
    if not isinstance(prompt, str) or len(prompt) > 1024:
        raise ValueError("One text prompt of at most1024 characters required")
    text = SYSTEM_TEMPLATE + f"<|im_start|>user\n{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=[text], padding=True, padding_side="left", return_tensors="pt")
    system_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    drop_idx = len(processor.apply_chat_template(system_message, tokenize=True, return_dict=False)[0])
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return inputs, drop_idx, image_token_id


def tensor_fingerprint(tensor):
    import torch
    assert torch.isfinite(tensor).all()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
        "sha256_float32": hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "finite": True}


def denormalize_latents(latents, arch):
    import torch
    mean = torch.tensor(arch.latents_mean, dtype=latents.dtype, device=latents.device).view(1, arch.z_dim, 1, 1, 1)
    std = torch.tensor(arch.latents_std, dtype=latents.dtype, device=latents.device).view(1, arch.z_dim, 1, 1, 1)
    # Preserve upstream get_decode_scale_and_shift + DecodingStage operation
    # order: BF16 reciprocal first, divide, then add. Multiplication by std
    # changes BF16 rounding and is deliberately not substituted.
    scale = std.reciprocal()
    latents = latents / scale
    return latents + mean


def run_t2i(*, text_encoder, dit, vae, processor, scheduler, sigmas, prompt, height, width, seed, progress=None):
    """Connect genuine components. No component implementation is replaced here.

    T2I only, batch1, TP1, SP1, true CFG1, eager. Conditioning-cache/offload/graph
    orchestration is absent; genuine model prefix K/V caching follows upstream.
    The caller supplies loaded models and config; this function loads no weights.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
    assert list(sigmas) == SIGMAS
    assert height >= 32 and width >= 32 and height % 32 == width % 32 == 0
    device = next(dit.parameters()).device
    dit_dtype = next(dit.parameters()).dtype
    assert next(text_encoder.parameters()).device == next(vae.parameters()).device == device
    text_encoder.model.visual.fp32_position_interpolation = False
    text_encoder.model.visual.rotary_pos_emb.recompute_on_device_change = True
    inputs, drop_idx, image_token_id = prompt_inputs(processor, prompt)
    inputs = inputs.to(device)
    captured = []
    norm = text_encoder.model.language_model.norm
    handle = norm.register_forward_pre_hook(lambda module, values: captured.append(values[0].detach()))
    try:
        with torch.inference_mode(), set_forward_context(current_timestep=None, attn_metadata=None):
            output = text_encoder(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        final_hidden = output.hidden_states[-1]
        assert len(captured) == 1 and torch.equal(final_hidden, captured[0]), "Conditioner must return pre-final-norm"
        if progress:
            progress("text_encoder_forward_completed", device=str(final_hidden.device))
    finally:
        handle.remove()
    valid = inputs.attention_mask[0].bool()
    hidden = final_hidden[0, valid][drop_idx:]
    ids = inputs.input_ids[0, valid][drop_idx:]
    hidden, slots = collapse_image_slots(hidden, ids, image_token_id)
    assert hidden.shape[0] > 0 and not slots.any(), "T2I entry forbids reference-image slots"
    layout = build_layout(slots.tolist(), [(1, height // 16, width // 16)], dit.config.axes_dims_rope, device)
    conditioning = hidden[None].to(dtype=dit_dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = randn_tensor((1, 1, dit.config.in_channels, height // 16, width // 16),
        generator=generator, device=device, dtype=dit_dtype)
    latents = latents.reshape(1, dit.config.in_channels, -1).transpose(1, 2)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    sched = scheduler.config
    length = (height // 16) * (width // 16)
    base_length, max_length = sched.get("base_image_seq_len", 256), sched.get("max_image_seq_len", 4096)
    slope = (sched.get("max_shift", 1.15) - sched.get("base_shift", .5)) / (max_length - base_length)
    mu = length * slope + (sched.get("base_shift", .5) - base_length * slope)
    scheduler.set_timesteps(sigmas=list(sigmas), device=device, mu=mu)
    assert len(scheduler.timesteps) == 8 and len(scheduler.sigmas) == 9
    caches = [[{} for _ in range(dit.config.num_layers)]]
    steps = []
    with torch.inference_mode():
        for index, timestep in enumerate(scheduler.timesteps):
            started = time.monotonic()
            with set_forward_context(current_timestep=index, attn_metadata=None):
                noise = dit(hidden_states=latents, encoder_hidden_states=conditioning,
                    timestep=timestep.expand(1).to(dtype=dit_dtype), layouts=[layout], prefix_caches=caches)
            assert noise.shape == latents.shape
            latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]
            assert all(cache.get("key") is not None and cache.get("value") is not None for cache in caches[0])
            steps.append({"index": index, "timestep": float(timestep), "elapsed_seconds": time.monotonic() - started,
                "latents": tensor_fingerprint(latents)})
            if progress:
                progress("DiT_scheduler_step_completed", device=str(latents.device), index=index)
        unpacked = latents.transpose(1, 2).reshape(1, -1, 1, height // 16, width // 16)
        ac = vae.config.arch_config
        vae_dtype = next(vae.parameters()).dtype
        denormalized = denormalize_latents(unpacked, ac)
        decoded = vae.decode(denormalized.to(dtype=vae_dtype))
        if progress:
            progress("VAE_decode_completed", device=str(decoded.device))
    assert decoded.shape == (1, 4, 1, height, width)
    receipt = {"prompt_ids": inputs.input_ids.tolist(), "drop_system_tokens": drop_idx,
        "conditioning": tensor_fingerprint(conditioning), "pre_final_norm_verified": True,
        "sigmas": scheduler.sigmas.tolist(), "steps": steps, "decoded": tensor_fingerprint(decoded),
        "prefix_cache_layers_populated": len(caches[0]), "condition_reference_images": 0}
    return decoded, receipt
