"""Build immutable probe007 from006, retaining every old artifact."""
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


def add_keywords(text, function_names, keywords):
    tree = ast.parse(text)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in function_names:
            lines = text.splitlines(True)
            closing = lines[node.end_lineno - 1]
            indent = len(closing) - len(closing.lstrip()) + 4
            edits.append((node.end_lineno - 1, " " * indent + keywords + ",\n"))
    lines = text.splitlines(True)
    for index, raw in sorted(edits, reverse=True):
        lines.insert(index, raw)
    return "".join(lines)


def main():
    source = ROOT / "candidates/qwen21-turbo-das007-006"
    dest = ROOT / "candidates/qwen21-turbo-das007-007"
    if dest.exists():
        raise ValueError("Fresh007 destination required")
    shutil.copytree(source, dest)
    path = dest / "probe_qwen21_core007.py"
    text = path.read_text()
    old = "                    spec.submodule_search_locations.insert(0, str(local))"
    new = "                    spec.submodule_search_locations = [str(local), *list(spec.submodule_search_locations)]"
    assert old in text
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    # Existing SRT vision implementations expose TP=1 through their real
    # use_data_parallel switch. No synthetic process group is introduced.
    path = dest / "overlay/sglang/multimodal_gen/runtime/models/encoders/qwen3vl_vision.py"
    text = add_keywords(path.read_text(), {"Qwen3_VisionMLP", "Qwen3VLMoeVisionPatchMerger"}, "use_data_parallel=True")
    path.write_text(text, encoding="utf-8")
    path = dest / "overlay/sglang/multimodal_gen/runtime/models/encoders/qwen_vl_vision.py"
    text = add_keywords(path.read_text(), {"QKVParallelLinear", "RowParallelLinear"}, "tp_size=1, tp_rank=0")
    text = text.replace("        self.num_heads = config.num_heads // get_tp_world_size()",
                        "        if get_tp_world_size() != 1:\n            raise ValueError(\"DAS probe profile requires TP=1\")\n        self.num_heads = config.num_heads")
    path.write_text(text, encoding="utf-8")
    lock = json.loads((dest / "source-lock.json").read_text())
    lock["port_changes"].extend([
        "007 repairs the source finder after actual005 raw traceback: namespace search locations are converted to list before prepend, preserving native initializer and paths.",
        "007 combines006 pure-text DiT input fix; this fixes the probe input, not model implementation.",
        "007 selects real old SRT vision use_data_parallel=True / tp_size=1,tp_rank=0 under strict TP1, preventing separate attention-TP global-group dependency."
    ])
    lock["actual005_raw_archive_sha256"] = "d80f5ed745b8a53c638eade8c8e8ee695d63a2947c3ba244d40e87ef1c15b7c3"
    lock["actual005_result"] = {"native_sources_verified_per_worker": 1989,
        "native_unchanged_workers": 5, "cuda_context_initialized_workers": 0,
        "component_loading_reached": False,
        "blocking_error": "AttributeError: '_NamespacePath' object has no attribute 'insert'"}
    archive = zipfile.ZipFile(ROOT / "dist/sglang-qwen21-efc984-source007.zip")
    prefix = archive.namelist()[0] + "python/"
    patches = []
    for row in lock["files"]:
        current = (dest / row["path"]).read_bytes()
        row["bytes"], row["sha256"] = len(current), sha(current)
        relative = row["path"].removeprefix("overlay/")
        original = archive.read(prefix + relative)
        row["upstream_unchanged"] = current == original
        ast.parse(current.decode("utf-8"), feature_version=(3, 10))
        if current != original:
            patches.extend(difflib.unified_diff(original.decode().splitlines(True), current.decode().splitlines(True),
                           fromfile="a/" + relative, tofile="b/" + relative))
    (dest / "source-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (dest / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
    shutil.copyfile(ROOT / "dist/000-session007-qwen21-core005-raw.zip", dest / "actual005-raw.zip")
    shutil.copyfile(ROOT / "evidence/session007/qwen21-core005-independent-audit.json", dest / "actual005-independent-audit.json")
    shutil.copyfile(Path(__file__), dest / "revise007-source.py")
    path = dest / "README.md"
    text = path.read_text().replace("007-006", "007-007")
    text += """
## Actual005 evidence and007 correction

All five005 workers verified 1,989 native Python sources before and after.
The native DAS source remained unchanged; CUDA context initialized was false.
Every worker stopped in the source finder on `_NamespacePath.insert`, before
component loading. That is an import-harness failure, with no model API conclusion.
Raw005 and its independent CRC/manifest audit are retained here.

007 converts actual namespace search paths to a list before prepend. It also
omits condition_latents for the pure-text DiT test and selects genuine native
vision TP1 constructor arguments. Dynamic007 imports/forwards remain pending.
"""
    path.write_text(text, encoding="utf-8")
    rows = []
    for path in sorted(dest.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            raw = path.read_bytes()
            rows.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-probe-bundle-v1", "files": rows}, indent=2) + "\n", encoding="utf-8")
    zip_path = ROOT / "dist/session007-qwen21-turbo-core-probe007.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
    archive_sha = sha(zip_path.read_bytes())
    helper = (ROOT / "scripts/unpack_qwen21_core_probe007_006.py").read_text().replace(
        "e56ce41eb9e676d33169fa97ec96c28fc7938822a3cdb52bede7eb03d61f3669", archive_sha)
    (ROOT / "scripts/unpack_qwen21_core_probe007_007.py").write_text(helper, encoding="utf-8")
    result = {"path": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": archive_sha,
              "manifest_payloads": len(rows), "overlay_modules": len(lock["files"]),
              "actual005_audited": True, "dynamic007_verified": False}
    with (ROOT / "evidence/session007/qwen21-probe007-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
