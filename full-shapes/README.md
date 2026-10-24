# Full native Qwen21 architecture and actual checkpoint-header probe012

This CPU/meta probe constructs the exact full text encoder, DiT and RGBA VAE
architectures from fixed model metadata, then compares every target state key
and shape with actual published safetensors headers. These headers came from
strict HTTP206 byte ranges:155,976 header bytes, zero requested tensor payload.
Source commits, exact expected whole-file sizes/SHA and header SHA are bound in
headers/header-receipt012.json. Headers are not full-file integrity proof.

The checkpoint has750 text parameters,297 DiT parameters across two shards, and
238 VAE parameters, all BF16. The one actual Qwen3VL visual qkv→qkv_proj name
conversion is preserved. Every source maps one-to-one to a target, with full key
and shape equality; no dropped/merged/duplicate source or missing-key synthesis
is permitted. Custom native parameter loaders are counted and retained for
the subsequent actual streaming load. Models stay on meta; no weights are loaded.

```bash
/root/private_data/guoxin-wan-env-006-001/venv/bin/python -B /root/private_data/qwen21-turbo-full-shapes012/probe_qwen21_full_shapes012.py --bundle /root/private_data/qwen21-turbo-core-probe010 --expected-bundle-sha256 73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d --output /root/private_data/qwen21-full-shapes012-results
```

The source bundle must have passed its core probes first. Native Linux CPython3.10
is required; all1989 native sources are hashed before/after, and output is new.
Imports may initialize a CUDA context. Full architectures use no tensor payload
and perform no GPU tensor compute. The probe lists meta nonpersistent buffers
so subsequent streaming load can materialize them through original native code.

qwen21_pretrained012.py also supplies a readonly safe_open iterator and genuine
native load_model_from_streaming_state_dict adapter. It requires a successful
bijective header plan and complete whole-file SHA verification by its caller;
this package's probe does not execute that GPU loader. Native custom parameter
weight_loader functions are preserved. Only known positional buffers may be
rebuilt by the original functions/classes; unknown meta buffers fail closed.
Actual full loading/inference remains pending. No official evaluation or
submission is claimed. Existing native/source/wheels remain unchanged.
