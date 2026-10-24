# Complete Qwen21 Turbo pretrained HCU runner014

014 supersedes unexecuted013 for full pretrained inference. The original013
package remains frozen. All original17 file whole-SHA checks, full architecture
header/key/shape bijection, genuine native streaming/custom parameter loaders,
HIP TP1/SP1, Torch SDPA, disabled optimization gates, source hashes and cleanup
are retained. One load generates256 then512 (or explicitly256 then1024).

## Corrections and additional verification

The decoder now preserves the exact original arithmetic: publisher std is
converted to the latent dtype, reciprocal is computed, latents are divided by it,
then mean is added. Multiplying directly by std is not equivalent in BF16.
Real Torch2.10 CPU ran the unmodified original scale_and_shift_latents function
from exact upstream against014: exact equality for float32/BF16. The old multiply
path differed on84/512 BF16 elements with max abs0.0625. Pixel normalization also
follows original `(image / 2 + .5).clamp(0,1)` before converting to float32.

After each native component load and before post-load packing, at most four
bounded source first/last and native-custom-loader first/last parameters are
compared with the actual GPU-loaded parameter. Each selected tensor is at most
32MiB. dtype/shape/torch.equal and exact BF16 byte SHA must agree. These actual
value checks complement the complete whole-file SHA and key/shape plan.

The real installed RocmPlatform is selected explicitly after actual HIP is
confirmed, avoiding vendor-management autodetection ambiguity. A real exact
QwenImage21PipelineConfig with the publisher sigmas is passed to real MM args;
SRT keeps its real separate auxiliary args. No platform or model is faked.
Stage progress records completed actual encoder, DiT and VAE GPU forwards. A
partial exception before a completed forward uses GPU_tensor_compute=null,
not an unsupported claim that no GPU work occurred.

## Native execution prerequisites

Complete17 checkpoint files must already be downloaded. A successful matching
full-shapes012.json is mandatory. Use the actual immutable source bundle and its
real manifest SHA. The source bundle must also pass native core forward probes.

```bash
/root/private_data/guoxin-wan-env-006-001/venv/bin/python -B /root/private_data/qwen21-turbo-pretrained014/run_qwen21_pretrained014.py --bundle /root/private_data/qwen21-turbo-core-probe010 --expected-bundle-sha256 73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d --shape-receipt /root/private_data/qwen21-full-shapes012-results/full-shapes012.json --resolutions 256,512 --output /root/private_data/qwen21-complete-pretrained014-results --max-seconds 1200
```

Output must be fresh. No dependency installation, native/source/checkpoint
modification, process takeover, HTTP service or submission is performed. The
actual GPU run remains pending. These are development images, with no official
quality, accuracy, performance or acceptance claim. Existing Apache, vendor and
Qwen RESEARCH notices are retained. No global relicensing applies.
