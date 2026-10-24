"""Freeze008 source and executable probe;007 raw audit is separate and pending."""
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
    source = ROOT / "revisions/qwen21-core008-draft"
    dest = ROOT / "candidates/qwen21-turbo-das007-008"
    if dest.exists():
        raise ValueError("Fresh008 candidate required")
    shutil.copytree(source, dest)
    lock = json.loads((dest / "source-lock.json").read_text())
    archive_path = ROOT / "dist/sglang-qwen21-efc984-source007.zip"
    assert sha(archive_path.read_bytes()) == lock["upstream_source_archive_sha256"]
    tree = json.loads((ROOT / "candidates/qwen21-turbo-das007-001/upstream-tree.json").read_text())
    blobs = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    records, patches = [], []
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        prefix = archive.namelist()[0] + "python/"
        for path in sorted((dest / "overlay").rglob("*")):
            if not path.is_file():
                continue
            assert not path.is_symlink() and "__pycache__" not in path.parts
            relative = path.relative_to(dest / "overlay").as_posix()
            original = archive.read(prefix + relative)
            raw = path.read_bytes()
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
    lock["files"] = records
    lock["profile"]["conditioning_cache"] = False
    lock["port_changes"].extend([
        "008: copy exact layernorm_modulate_triton/residual_gate_add_jit dynamic export owners plus seven real dependency modules and original residual_gate_add.cuh.",
        "008: remove optional cached_conditioning decorator/import; no active cache profile executes the same original function directly, as upstream cache=None branch does.",
        "008: native PatchEmbed constructor then proj.enable_linear=False preserves upstream disable_linear=True Conv3d path.",
        "008: RMSNorm adapter selects native _forward_method, matching upstream force_native=True without overriding MultiPlatformOp.forward."
    ])
    lock["actual007_raw_independently_audited"] = False
    lock["coordinator_report007"] = {"source": "root message; original archive pending local arrival",
        "reported_passed": ["config", "scheduler_processor", "vae"],
        "reported_failures": {"dit": "missing dynamic KDA layernorm module", "encoder": "optional conditioning-cache import requires absent get_replica_group"}}
    (dest / "source-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (dest / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
    shutil.copyfile(Path(__file__), dest / "seal008-source.py")
    shutil.copyfile(ROOT / "scripts/resolve_qwen21_core008.py", dest / "resolve008-source.py")
    path = dest / "README.md"
    text = path.read_text().replace("core-probe007-results", "core-probe008-results")
    text += """
## Executable008 correction

008 binds the actual dynamic KDA kernel owners and transitive source dependencies.
It executes the original encoder image-features method without the optional
conditioning-cache decorator, whose no-active-cache branch directly calls that
same original method. TP1, no cache, eager CPU bounds remain explicit.
It also uses native Conv3dLayer.enable_linear=False and native RMSNorm dispatch.
No framework stub, fake get_replica_group or replacement math is introduced.

Root reports007 config, scheduler/processor and VAE passed; that original archive
has not yet arrived locally at008 seal time, so this is recorded as a coordinator
report, not an independent raw audit. Actual008 imports/forwards remain pending.
The original005 audit is retained. Neither GPU/full checkpoint nor service
readiness is claimed. Original005/006/007 ZIPs and native wheel are unchanged.
"""
    path.write_text(text, encoding="utf-8")
    (dest / "DRAFT008.json").rename(dest / "008-preparation-history.json")
    files = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            files.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-probe-bundle-v1", "files": files}, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "dist/session007-qwen21-turbo-core-probe008.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    archive_sha = sha(archive_path.read_bytes())
    text = (ROOT / "scripts/unpack_qwen21_core_probe007_007.py").read_text().replace(
        "664475489928be75d5ce0dc7dc00e4f0dedbbebf702a1464cfd254d287692c3e", archive_sha)
    (ROOT / "scripts/unpack_qwen21_core_probe007_008.py").write_text(text, encoding="utf-8")
    result = {"path": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": archive_sha,
        "payload_files": len(files), "upstream_source_files": len(records),
        "Python_modules": sum(row["path"].endswith(".py") for row in records),
        "actual007_independent_raw_audit": False, "dynamic008_verified": False,
        "native_modified": False, "GPU_executed": False}
    with (ROOT / "evidence/session007/qwen21-probe008-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
