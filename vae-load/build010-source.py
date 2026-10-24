"""Package a readonly real-checkpoint VAE CPU probe against sealed009."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    dest = ROOT / "tasks/qwen21-turbo-vae-load010"
    dest.mkdir(exist_ok=False)
    for source in (ROOT / "scripts/probe_qwen21_vae_checkpoint010.py",
        ROOT / "evidence/session007/qwen21-pipeline-prep009/download-sources009.json"):
        shutil.copyfile(source, dest / source.name)
        if source.suffix == ".py":
            ast.parse(source.read_text(), feature_version=(3, 10))
    (dest / "README.md").write_text('''# Actual Qwen21 VAE checkpoint CPU probe 010

Run after the009 metadata + VAE phase has14 full-SHA verified files. This probe
strictly binds the sealed009 source bundle manifest and fixed transfer lock,
re-hashes all14 actual files, uses the original complete VAE architecture,
constructs its real SGLang model on meta, and compares every checkpoint key and
shape before `load_state_dict(strict=True, assign=True)`. Missing/unexpected keys
or shapes abort. It then runs32×32 RGBA encode/decode on CPU float32 and records
parameter count, actual dtype/key/shape fingerprints and finite tensor hashes.

```bash
/root/private_data/guoxin-wan-env-006-001/venv/bin/python -B /root/private_data/qwen21-turbo-vae-load010/probe_qwen21_vae_checkpoint010.py --bundle /root/private_data/qwen21-turbo-core-probe009 --output /root/private_data/qwen21-vae-checkpoint010-results --timeout-seconds 300
```

Output must be new and outside source/checkpoint directories. The exact Qwen
checkpoint namespace is read-only. All1,989 native sources are verified before
and after. Imports can initialize a CUDA context; tensors and all compute target
CPU. A Unix alarm requests exit after the bounded timeout; a coordinating parent
should enforce its own process timeout if a native call stops returning.

The original native VAE loader uses strict=False. This dedicated probe instead
rejects every missing/unexpected key or shape and never fills absent parameters.
It does not load text encoder or DiT weights, generate a semantic image, or claim
full-pipeline, GPU, official accuracy, performance, acceptance or submission.
010 is locally Python3.10 syntax checked. Actual native execution is pending.
''', encoding="utf-8")
    shutil.copyfile(Path(__file__), dest / "build010-source.py")
    rows = []
    for path in sorted(dest.iterdir()):
        raw = path.read_bytes()
        rows.append({"path": path.name, "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-checkpoint-probe-v1", "files": rows}, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "dist/session007-qwen21-vae-load010.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    value = sha(archive_path.read_bytes())
    helper = ROOT / "scripts/unpack_qwen21_vae_load010.py"
    text = (ROOT / "scripts/unpack_qwen21_core_probe007_009.py").read_text().replace(
        "c918dd034d6e807deab79a10dd9599871aee0f0370ebdda1ccba742877c2ca50", value)
    helper.write_text(text, encoding="utf-8")
    result = {"zip": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": value,
        "helper": str(helper), "helper_sha256": sha(helper.read_bytes()), "payload_files": len(rows),
        "actual_checkpoint_load_executed": False, "GPU_executed": False}
    with (ROOT / "evidence/session007/qwen21-vae-load010-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
