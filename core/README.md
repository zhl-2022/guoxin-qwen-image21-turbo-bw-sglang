# Native Qwen21 core010 and executable five-component chain011

Core010 fixes the two original009 runtime interfaces. DiT now gets a model-local
subclass of the genuine native USPAttention. It copies the exact upstream SP1
branch: concatenate prefix/suffix K/V and call real native attn_impl.forward with
the real forward context. SP>1 is rejected. The model math is unchanged and no
new NVIDIA JIT kernel is invoked. The CPU profile selects genuine Torch SDPA.

Encoder native RoPE reads the separate SRT global args. Core010 creates a real
native SRT ServerArgs in its own supported no-serving model_path="None" mode and
publishes it through the real setter before constructing RoPE. This config object
does not serve a model; actual Qwen config/source/checkpoint binding remains in
the real MM model path. MM ServerArgs is also real. No fake framework module or
object is synthesized. Native files and wheel remain unchanged.

## Core execution

```bash
python -B probe_qwen21_core007.py --bundle . --components dit encoder --output /root/private_data/qwen21-core-probe010-results
```

The original009 archive passed CRC/all5 payloads and frozen bundle binding; two
workers each verified1989 native sources before/after. DiT reached actual forward
and failed only on missing replicated-prefix API. Encoder reached actual text
RoPE construction and failed on uninitialized SRT args. Neither was a numerical
failure. All699 recorded native/overlay module hashes match fixed source bytes.
The actual010 native results remain pending.

## Single-request five-component chain011

The included probe_qwen21_pipeline011.py and qwen21_t2i_stage011.py connect actual
Qwen3VLProcessor, genuine native Qwen3VL encoder, DiT, RGBA VAE and real Diffusers
scheduler as an8-sigma T2I chain. It uses small random architectures and records
that scope explicitly. A real hook asserts the encoder conditioning equals the
pre-final-norm input. The original system-template/drop14, sigma8, prefix caches,
latent pack/unpack and decode mean/std are retained. It produces a32×32 RGBA PNG
as wiring evidence only. The stage function accepts actual already loaded model
objects and will also be reused for strict pretrained loading; it synthesizes no
component implementations. Dynamic native execution is pending.

The CLI requires an explicit --expected-bundle-sha256 equal to this package's
MANIFEST.json SHA256, plus --bundle and a fresh --output directory. Execute only
after core010 passes; native Linux CPython3.10 DAS is required. Local real HF5.5
processor and Diffusers0.37 checks verified template/drop14, eight timesteps,
upstream collapse_image_slots AST identity, exact pack/unpack, and float32 latent
stats roundtrip (max abs5.3644e-7). This local evidence is not a native-chain pass.

## Identity and copyright

Fixed upstream efc9849520216b31b8a4a4322399e47e8dbde067, model revision
fc13ba1986fcb905854dc8de2f52eee395f49b5b, native DAS wheel SHA256
7c13be91a56b160af81847fffbf4f177b0f07fe0717b4042960ee23846ae7d61.
The package has201 Python overlay modules plus291 other real source/header
assets, all Git-blob/SHA bound. All original notices, vendor licenses and Qwen
RESEARCH license remain. No global relicensing, complete-checkpoint loading,
BW tensor compute, official accuracy/performance/acceptance/submission is claimed.
VAE actual-checkpoint probe010 is a separate package named vae-load010.
