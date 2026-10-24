"""Freeze a new009 overlay from immutable008 and the exact upstream archive."""
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
    dest = ROOT / "candidates/qwen21-turbo-das007-009"
    assert not dest.exists(), "009 must be a fresh immutable candidate"
    source = ROOT / "candidates/qwen21-turbo-das007-008"
    shutil.copytree(source, dest)
    (dest / "MANIFEST.json").unlink()
    lock = json.loads((dest / "source-lock.json").read_text())
    archive_path = ROOT / "dist/sglang-qwen21-efc984-source007.zip"
    assert sha(archive_path.read_bytes()) == lock["upstream_source_archive_sha256"]
    tree = json.loads((ROOT / "candidates/qwen21-turbo-das007-001/upstream-tree.json").read_text())
    blobs = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        prefix = archive.namelist()[0] + "python/"
        assets = []
        for name in archive.namelist():
            if name.endswith("/") or not any(name.startswith(prefix + item) for item in (
                    "sglang/kernels/jit/include/", "sglang/kernels/jit/csrc/")):
                continue
            relative = name[len(prefix):]
            target = dest / "overlay" / relative
            assert not target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            assets.append(relative)
        assert len(assets) == 291
        base_raw = archive.read(prefix + "sglang/multimodal_gen/runtime/models/encoders/base.py")
        base = next(node for node in ast.parse(base_raw).body if isinstance(node, ast.ClassDef) and node.name == "TextEncoder")
        assignment = next(node for node in base.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "layer_names" for target in node.targets))
        layer_names = ast.literal_eval(assignment.value)
        assert layer_names == ["layers", "encoder.block", "text_model.encoder.layers", "model.language_model.layers"]
        encoder = dest / "overlay/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py"
        text = encoder.read_text()
        assert text.count("        *TextEncoder.layer_names,") == 1
        text = text.replace("        *TextEncoder.layer_names,", "        # Exact upstream TextEncoder metadata; native DAS base predates it.\n" +
            "\n".join("        " + repr(name) + "," for name in layer_names))
        encoder.write_text(text, encoding="utf-8")
        records, patches = [], []
        for path in sorted((dest / "overlay").rglob("*")):
            if not path.is_file():
                continue
            assert not path.is_symlink() and "__pycache__" not in path.parts
            relative = path.relative_to(dest / "overlay").as_posix()
            original, raw = archive.read(prefix + relative), path.read_bytes()
            blob = hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
            assert blob == blobs["python/" + relative]
            if path.suffix == ".py":
                ast.parse(raw.decode(), feature_version=(3, 10))
            records.append({"path": "overlay/" + relative, "bytes": len(raw), "sha256": sha(raw),
                "upstream_sha256": sha(original), "upstream_git_blob_sha1": blob,
                "upstream_unchanged": raw == original})
            if raw != original:
                patches.extend(difflib.unified_diff(original.decode().splitlines(True), raw.decode().splitlines(True),
                    fromfile="a/" + relative, tofile="b/" + relative))
    probe = dest / "probe_qwen21_core007.py"
    text = probe.read_text()
    needle = '    args = parser.parse_args()\n'
    assert text.count(needle) == 1
    text = text.replace(needle, '    parser.add_argument("--components", nargs="+", choices=["scheduler_processor", "config", "vae", "dit", "encoder"],\n'
        '                        default=["scheduler_processor", "config", "vae", "dit", "encoder"])\n' + needle)
    text = text.replace('    for name in ("scheduler_processor", "config", "vae", "dit", "encoder"):',
                        '    for name in dict.fromkeys(args.components):')
    text = text.replace('"passed": all(item.get("returncode") == 0 for item in workers),',
        '"passed": all(item.get("returncode") == 0 for item in workers),\n'
        '               "components_selected": list(dict.fromkeys(args.components)),\n'
        '               "all_five_components_selected": len(set(args.components)) == 5,')
    ast.parse(text, feature_version=(3, 10))
    probe.write_text(text, encoding="utf-8")
    lock["files"] = records
    lock["port_changes"].extend([
        "009: include all291 exact upstream JIT include/csrc files and vendor license, satisfying unchanged paths.py discovery without fake directories or path overrides.",
        "009: inline exact four upstream TextEncoder.layer_names in concrete Qwen3VL class; native base remains unchanged and TP1 no-offload math is unaffected."])
    lock["actual008_raw_independently_audited"] = True
    lock["actual008_raw_sha256"] = "4f03d8071176d09cd002d5676d308ca5d2c9081a0f8f92e548799c2bd719ddfb"
    (dest / "source-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (dest / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
    for name in ("qwen21-core008-independent-audit.json", "qwen21-core008-loaded-source-bindings.json"):
        shutil.copyfile(ROOT / "evidence/session007" / name, dest / name)
    shutil.copyfile(Path(__file__), dest / "build009-source.py")
    (dest / "README.md").write_text('''# Qwen-Image 2.1 Turbo native DAS core probe 009

This isolated candidate binds 200 Python modules and 292 source/header assets to
upstream `efc9849520216b31b8a4a4322399e47e8dbde067`, model revision
`fc13ba1986fcb905854dc8de2f52eee395f49b5b`, and unchanged DAS SGLang 0.5.12 wheel
SHA256 `7c13be91a56b160af81847fffbf4f177b0f07fe0717b4042960ee23846ae7d61`.

## Actual008 evidence

The original008 archive passed CRC, all11 payload hashes and frozen bundle binding.
Scheduler/processor, config and tiny RGBA VAE passed. DiT stopped in unchanged
JIT paths.py because source assets were absent; encoder stopped in class definition
because native TextEncoder has no layer_names. Five workers each verified all1,989
native Python files before and after. Native imports initialized CUDA contexts;
there was no GPU tensor compute, checkpoint loading or full service evidence.

## Corrections and execution

009 supplies the complete exact upstream JIT include/csrc assets and their vendor
license. Unchanged path discovery now sees real source directories. The concrete
Qwen class includes the same four layer paths as upstream TextEncoder instead of
reading a missing old-base attribute. Native base, framework and kernels stay
unchanged. The previous native RMSNorm dispatch, Conv3d non-linear projection,
namespace finder and no-active-conditioning-cache corrections are preserved.

```bash
python -B probe_qwen21_core007.py --bundle . --audit-only
python -B probe_qwen21_core007.py --bundle . --components dit encoder --output /root/private_data/qwen21-core-probe009-results
```

Dynamic execution requires the actual Linux CPython3.10 native DAS environment.
Without --components the original five workers run. Selection is recorded, so
two successful selected workers cannot establish all-five validation. Workers use
real CpuPlatform, TP1/SP1 Gloo, Torch SDPA and eager float32 tiny models; each has
a150-second process timeout. Output must be fresh and outside this bundle.

## Bounds and copyright

No stub framework, fake JIT path, checkpoint weight or weight downloader is used.
No compile, graph, quantization, offload, parallel VAE, official accuracy,
performance, acceptance or submission is claimed. Real native imports and tiny
forwards for009 await original runtime evidence. Compatibility.patch records
every source edit; source-lock binds every exact original and derived byte.
Original Apache notices and vendor licenses are retained. The model's Qwen
RESEARCH license remains in model-assets/LICENSE. No global relicensing applies.
''', encoding="utf-8")
    files = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            files.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-probe-bundle-v1", "files": files}, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "dist/session007-qwen21-turbo-core-probe009.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    archive_sha = sha(archive_path.read_bytes())
    helper = ROOT / "scripts/unpack_qwen21_core_probe007_009.py"
    helper.write_text((ROOT / "scripts/unpack_qwen21_core_probe007_008.py").read_text().replace(
        "4eda753c3fd8c26e9c7542561a0086320c64293872ad1835809c25cdbc133046", archive_sha), encoding="utf-8")
    result = {"path": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": archive_sha,
        "helper": str(helper), "helper_sha256": sha(helper.read_bytes()), "payload_files": len(files),
        "upstream_source_files": len(records), "Python_modules": sum(row["path"].endswith(".py") for row in records),
        "actual008_independent_raw_audit": True, "dynamic009_verified": False,
        "native_modified": False, "GPU_executed": False}
    with (ROOT / "evidence/session007/qwen21-probe009-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
