"""Resolve actual007 import failures in the new unsealed008 draft only."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "revisions/qwen21-core008-draft"


def main():
    assert DEST.is_dir() and not (DEST / "MANIFEST.json").exists()
    path = DEST / "overlay/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py"
    text = path.read_text()
    imported = "from sglang.multimodal_gen.runtime.cache.conditioning import cached_conditioning\n"
    assert imported in text and "    @cached_conditioning\n" in text
    text = text.replace(imported, "# TP1 probe has no active conditioning cache: call the original method directly.\n")
    text = text.replace("    @cached_conditioning\n", "")
    path.write_text(text, encoding="utf-8")
    # The removed wrapper's own upstream no-active-cache branch is `return fn`.
    # Retain its exact source as evidence, but exclude this unsupported newer
    # request-cache module from the active serial-core overlay.
    cache = DEST / "overlay/sglang/multimodal_gen/runtime/cache/conditioning.py"
    (DEST / "upstream-no-cache-evidence.py").write_bytes(cache.read_bytes())
    cache.unlink()
    lock = json.loads((DEST / "source-lock.json").read_text())
    source_path = ROOT / "dist/sglang-qwen21-efc984-source007.zip"
    wheel_path = ROOT / "evidence/session006/wan-preflight/hygon/wheels" / lock["native_wheel"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == lock["upstream_source_archive_sha256"]
    assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == lock["native_wheel_sha256"]
    with zipfile.ZipFile(source_path) as archive, zipfile.ZipFile(wheel_path) as wheel:
        prefix = archive.namelist()[0] + "python/"
        upstream = {name[len(prefix):]: archive.read(name) for name in archive.namelist()
                    if name.startswith(prefix + "sglang/") and name.endswith(".py")}
        native = set(wheel.namelist())
        seeds = ["sglang/kernels/kda_kernels/layernorm_modulate_triton.py",
                 "sglang/kernels/kda_kernels/residual_gate_add_jit.py"]
        added = []
        def add(relative):
            target = DEST / "overlay" / relative
            if target.exists() or relative in native:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(upstream[relative])
            added.append(relative)
            queue.append(relative)
            for parent in PurePosixPath(relative).parents:
                if str(parent) in (".", "sglang"):
                    continue
                init = parent.as_posix() + "/__init__.py"
                if init in upstream:
                    add(init)
        queue = []
        for seed in seeds:
            add(seed)
        missing = []
        while queue:
            relative = queue.pop(0)
            tree = ast.parse((DEST / "overlay" / relative).read_text())
            module = relative[:-3].replace("/", ".")
            package = module.rsplit(".", 1)[0] if not relative.endswith("/__init__.py") else module[:-9]
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    name = node.module or ""
                    if node.level:
                        name = importlib.util.resolve_name("." * node.level + name, package)
                    requested = [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("sglang."):
                            options = [alias.name.replace(".", "/") + suffix for suffix in (".py", "/__init__.py")]
                            found = next((item for item in options if item in native or item in upstream), None)
                            if found:
                                add(found)
                    continue
                else:
                    continue
                if not name.startswith("sglang."):
                    continue
                options = [name.replace(".", "/") + suffix for suffix in (".py", "/__init__.py")]
                found = next((item for item in options if item in native or item in upstream), None)
                if found:
                    add(found)
                else:
                    missing.append({"source": relative, "line": node.lineno, "module": name})
                for child in requested:
                    for suffix in (".py", "/__init__.py"):
                        child_path = name.replace(".", "/") + "/" + child + suffix
                        if child_path in upstream:
                            add(child_path)
            if len(added) > 100:
                raise ValueError("Unexpected dependency expansion; inspect actual source")
    result = {"added_real_upstream_modules": added, "added_count": len(added), "missing_module_paths": missing,
              "conditioning_cache": False, "cache_semantics": "Execute upstream original get_image_features directly; its cache=None path returns that function unchanged",
              "native_modified": False, "GPU_executed": False, "sealed": False}
    with (DEST / "closure008.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
