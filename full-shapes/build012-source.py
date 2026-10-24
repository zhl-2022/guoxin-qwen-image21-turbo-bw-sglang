"""Seal actual tensor-header metadata and a full native architecture shape probe."""
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
    dest = ROOT / "tasks/qwen21-turbo-full-shapes012"
    dest.mkdir(exist_ok=False)
    for name in ("qwen21_pretrained012.py", "probe_qwen21_full_shapes012.py", "fetch_qwen21_safetensors_headers012.py"):
        raw = (ROOT / "scripts" / name).read_bytes()
        ast.parse(raw.decode(), feature_version=(3, 10))
        (dest / name).write_bytes(raw)
    shutil.copytree(ROOT / "evidence/session007/qwen21-safetensors-headers012", dest / "headers")
    shutil.copyfile(ROOT / "candidates/qwen21-turbo-das007-010/LICENSE-upstream-APACHE-2.0", dest / "LICENSE-upstream-APACHE-2.0")
    (dest / "README.md").write_text('''# Full native Qwen21 architecture and actual checkpoint-header probe012

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
''', encoding="utf-8")
    shutil.copyfile(Path(__file__), dest / "build012-source.py")
    rows = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            raw = path.read_bytes(); rows.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-full-shapes-v1", "files": rows}, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "dist/session007-qwen21-full-shapes012.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file(): archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive_path) as archive: assert archive.testzip() is None
    value = sha(archive_path.read_bytes())
    helper = ROOT / "scripts/unpack_qwen21_full_shapes012.py"
    helper.write_text((ROOT / "scripts/unpack_qwen21_core_probe007_010.py").read_text().replace(
        "970bf1270fa18d51ab247d9a096c0e2ca6ebb26075cb40b9f3d623592ff0eecb", value), encoding="utf-8")
    result = {"zip": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": value,
        "helper": str(helper), "helper_sha256": sha(helper.read_bytes()), "payloads": len(rows),
        "actual_native_full_architecture_probe_executed": False, "tensor_payload_bytes_requested": 0, "GPU_executed": False}
    with (ROOT / "evidence/session007/qwen21-full-shapes012-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
