"""Seal014 exact decoder order and actual loaded-parameter verification."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    dest = ROOT / "tasks/qwen21-turbo-pretrained014"
    dest.mkdir(exist_ok=False)
    draft = ROOT / "revisions/qwen21-pretrained014-draft2"
    originals = ROOT / "tasks/qwen21-turbo-pretrained013"
    patch = []
    for before, after in (("run_qwen21_pretrained013.py", "run_qwen21_pretrained014.py"),
            ("qwen21_pretrained012.py", "qwen21_pretrained014.py"), ("qwen21_t2i_stage011.py", "qwen21_t2i_stage014.py")):
        raw = (draft / after).read_bytes()
        ast.parse(raw.decode(), feature_version=(3, 10))
        (dest / after).write_bytes(raw)
        patch.extend(difflib.unified_diff((originals / before).read_text().splitlines(True), raw.decode().splitlines(True),
            fromfile="013/" + before, tofile="014/" + after))
    (dest / "013-to-014.patch").write_text("".join(patch), encoding="utf-8")
    shutil.copytree(originals / "headers", dest / "headers")
    for name in ("download-sources009.json", "LICENSE-upstream-APACHE-2.0", "LICENSE-model-Qwen-RESEARCH"):
        shutil.copyfile(originals / name, dest / name)
    shutil.copyfile(ROOT / "evidence/session007/qwen21-decode-order014-real-CPU-regression-b.json", dest / "decode-order-real-CPU-regression.json")
    shutil.copyfile(ROOT / "scripts/probe_qwen21_decode_order014.py", dest / "probe_qwen21_decode_order014.py")
    shutil.copyfile(Path(__file__), dest / "seal014-source.py")
    (dest / "README.md").write_text('''# Complete Qwen21 Turbo pretrained HCU runner014

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
''', encoding="utf-8")
    rows = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            raw = path.read_bytes(); rows.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-pretrained-runner-v1", "files": rows}, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "dist/session007-qwen21-pretrained014.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file(): archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive_path) as archive: assert archive.testzip() is None
    value = sha(archive_path.read_bytes())
    helper = ROOT / "scripts/unpack_qwen21_pretrained014.py"
    helper.write_text((ROOT / "scripts/unpack_qwen21_pretrained013.py").read_text().replace(
        "77be3a378abb0486f1b022d09194d6185e682ceb79984d902ccbb50fbd7df247", value), encoding="utf-8")
    result = {"zip": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": value,
        "helper": str(helper), "helper_sha256": sha(helper.read_bytes()), "payloads": len(rows),
        "actual_GPU_run_executed": False, "checkpoint_tensor_payload_included": False,
        "actual_CPU_decode_order_regression_passed": True}
    with (ROOT / "evidence/session007/qwen21-pretrained014-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
