"""Resolve real Qwen21 module additions against DAS0512 into an isolated probe bundle.

Copies real immutable upstream implementations, never synthetic modules. Moves
two import paths to their existing DAS0512 implementations. No model/GPU load.
"""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "candidates/qwen21-turbo-das007-001"
DEST = ROOT / "candidates/qwen21-turbo-das007-005"
COMMIT = "efc9849520216b31b8a4a4322399e47e8dbde067"
WHEEL_SHA = "7c13be91a56b160af81847fffbf4f177b0f07fe0717b4042960ee23846ae7d61"
MOVES = {"sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload":
         "sglang.multimodal_gen.runtime.managers.layerwise_offload",
         "sglang.multimodal_gen.runtime.managers.memory_managers.component_manager":
         "sglang.multimodal_gen.runtime.managers.component_manager"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def top_names(raw):
    tree = ast.parse(raw.decode("utf-8"))
    names, wildcard = set(), False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            wildcard |= any(alias.name == "*" for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names, wildcard


def replace_node(text, node, replacement):
    lines = text.splitlines(True)
    lines[node.lineno - 1:node.end_lineno] = [replacement + "\n"]
    return "".join(lines)


def port_serial(path, raw):
    text = raw.decode("utf-8")
    if path.endswith("/models/dits/qwen_image21.py"):
        tree = ast.parse(text)
        for node in sorted(tree.body, key=lambda n: n.lineno, reverse=True):
            if isinstance(node, ast.ImportFrom) and node.module in {
                    "sglang.multimodal_gen.runtime.breakable_cuda_graph.replay_token",
                    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"}:
                text = replace_node(text, node, "")
        text = text.replace("@eager_on_graph\n", "")
        text = text.replace("        token = get_current_replay_token()\n        if self.value is None or token != self.token:\n            self.value, self.token = int(self.rows.item()), token\n        return self.value",
                            "        # Eager-only profile reads each invocation; no graph replay cache.\n        return int(self.rows.item())")
        text = text.replace("LayerwiseOffloadableModuleMixin", "OffloadableDiTMixin")
    if path.endswith("/models/vaes/autoencoder_kl_qwenimage21.py"):
        tree = ast.parse(text)
        for node in sorted(tree.body, key=lambda n: n.lineno, reverse=True):
            if isinstance(node, ast.ImportFrom) and node.module in {
                    "sglang.multimodal_gen.runtime.distributed",
                    "sglang.multimodal_gen.runtime.layers.parallel_conv"}:
                text = replace_node(text, node, "")
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
                    "QwenImage21SpatialConv3d", "enable_qwen21_spatial_decode"}:
                text = replace_node(text, node, "")
        text = text.replace("    can_install_spatial_shard_parallel_decode,\n    should_run_spatial_shard_parallel_decode,\n", "")
        text = text.replace("        *ParallelTiledVAE.layer_names,", "        \"encoder.down_blocks\", \"decoder.up_blocks\", \"encoder.down\", \"decoder.up\",")
        text = text.replace("        self.spatial_parallel = (\n            config.load_decoder and can_install_spatial_shard_parallel_decode(config)\n        )\n        if self.spatial_parallel:\n            enable_qwen21_spatial_decode(self.decoder)",
                            "        if config.use_parallel_decode or config.use_parallel_tiling:\n            raise ValueError(\"Probe profile requires serial VAE decode\")\n        self.spatial_parallel = False")
        start = text.index("        parallel = self.spatial_parallel and should_run_spatial_shard_parallel_decode(")
        end = text.index("        if self.config.patch_size is not None:", start)
        text = text[:start] + "        x = self.decoder(z, first_chunk=True)\n" + text[end:]
        text = text.replace("        if parallel:\n            x = gather_and_trim_height(x, expected_height)\n", "")
        text = text.replace("        if self.spatial_parallel:\n            x, heights = gather_variable_height(x)\n", "")
        text = text.replace("        return chunk_height_by_sizes(x, heights) if self.spatial_parallel else x", "        return x")
    if path.endswith("/models/encoders/qwen_vl_vision.py"):
        text = text.replace("from sglang.srt.runtime_context import get_parallel", "from sglang.multimodal_gen.runtime.distributed import get_tp_world_size")
        text = text.replace("        parallel = get_parallel()\n        self.num_heads = config.num_heads // parallel.tp_size", "        self.num_heads = config.num_heads // get_tp_world_size()")
        text = text.replace('            parallel_group="tp",\n', '')
        start = text.index("        backend = get_attn_backend(")
        end = text.index("    def _packed_attention(", start)
        text = text[:start] + "        # Serial probe uses the upstream packed-sequence Torch SDPA fallback.\n        self._attention_impl = None\n\n" + text[end:]
    if path.endswith("/configs/pipeline_configs/qwen_image21.py"):
        text = text.replace("vae_tiling: bool = field(default_factory=lambda: current_platform.is_gfx1151())", "vae_tiling: bool = False")
        text = text.replace("not self.vae_tiling and current_platform.is_gfx1151()", "False")
    if path.endswith("/models/encoders/qwen3vl_vision.py"):
        text = text.replace('            parallel_group="tp",\n', '').replace('                parallel_group="tp",\n', '')
    if path.endswith(("/models/dits/qwen_image21.py", "/models/encoders/qwen3vl.py")):
        # The baseline RMSNorm already exposes the exact native reduction;
        # select it through a model-local subclass rather than a new framework.
        marker = "from sglang.srt.layers.layernorm import RMSNorm\n"
        adapter = """

class Qwen21DASNativeRMSNorm(RMSNorm):
    \"Select the existing DAS native reduction used by upstream force_native.\"

    def __init__(self, *args, force_native=True, **kwargs):
        if not force_native:
            raise ValueError("Qwen21 profile requires native RMSNorm reduction")
        super().__init__(*args, **kwargs)

    def forward(self, x, residual=None, post_residual_addition=None):
        return self.forward_native(x, residual, post_residual_addition)

"""
        text = text.replace(marker, marker + adapter)
        text = text.replace("return RMSNorm(", "return Qwen21DASNativeRMSNorm(").replace("self.norm_q = RMSNorm(", "self.norm_q = Qwen21DASNativeRMSNorm(").replace("self.norm_k = RMSNorm(", "self.norm_k = Qwen21DASNativeRMSNorm(")
    return text.encode()


def main():
    if DEST.exists():
        raise ValueError("Fresh candidate destination required")
    upstream_archive = ROOT / "dist/sglang-qwen21-efc984-source007.zip"
    wheel = next((ROOT / "evidence/session006/wan-preflight/hygon/wheels").glob("sglang-0.5.12*.whl"))
    if sha(wheel.read_bytes()) != WHEEL_SHA:
        raise ValueError("DAS baseline wheel differs")
    with zipfile.ZipFile(upstream_archive) as current, zipfile.ZipFile(wheel) as baseline:
        if current.testzip() is not None or baseline.testzip() is not None:
            raise ValueError("Source/wheel CRC failure")
        prefix = current.namelist()[0]
        tree = json.loads((SOURCE / "upstream-tree.json").read_text())
        blobs = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
        official = {name[len(prefix + "python/"):]: current.read(name) for name in current.namelist()
                    if name.startswith(prefix + "python/sglang/") and name.endswith(".py")}
        native = {name: baseline.read(name) for name in baseline.namelist() if name.endswith(".py")}
        supplied = {row["path"][len("upstream/"):]: (SOURCE / row["path"]).read_bytes()
                    for row in json.loads((SOURCE / "source-lock.json").read_text())["source_files"]
                    if row["path"].startswith("upstream/")}
        # Phase-one core probes deliberately omit orchestration stages; the
        # native encoder/VAE bases are retained for a serial eager profile.
        supplied = {path: raw for path, raw in supplied.items()
                    if "/runtime/pipelines/" not in path and "/pipelines_core/" not in path}
        supplied = {path: port_serial(path, raw) for path, raw in supplied.items()}
        # Diffusion facade resolves exports dynamically. Include every real
        # diffusion implementation and metadata-only public operator facade.
        for path in official:
            if (path.startswith("sglang/kernels/ops/diffusion/")
                    or (path.startswith("sglang/kernels/ops/") and path.endswith("/__init__.py") and path.count("/") == 4)):
                supplied[path] = official[path]
        queue, scanned, missing_symbols, missing_modules = list(supplied), set(), [], []
        def add(path):
            if path not in supplied:
                supplied[path] = port_serial(path, official[path])
                queue.append(path)
            parent = PurePosixPath(path).parent
            while str(parent) not in (".", "sglang"):
                init = parent.as_posix() + "/__init__.py"
                if init not in native and init in official and init not in supplied:
                    supplied[init] = official[init]
                    queue.append(init)
                parent = parent.parent
        while queue:
            path = queue.pop(0)
            if path in scanned:
                continue
            scanned.add(path)
            text = supplied[path].decode("utf-8")
            for old, new in MOVES.items():
                text = text.replace(old, new)
            supplied[path] = text.encode()
            module = path[:-3].replace("/", ".")
            package = module.rsplit(".", 1)[0] if not path.endswith("/__init__.py") else module[:-9]
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                    if node.level:
                        import importlib.util
                        imported = importlib.util.resolve_name("." * node.level + imported, package)
                    requested = [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("sglang."):
                            candidates = [alias.name.replace(".", "/") + suffix for suffix in (".py", "/__init__.py")]
                            found = next((name for name in candidates if name in supplied or name in native or name in official), None)
                            if found and found not in native:
                                add(found)
                    continue
                else:
                    continue
                if not imported.startswith("sglang."):
                    continue
                candidates = [imported.replace(".", "/") + suffix for suffix in (".py", "/__init__.py")]
                found = next((name for name in candidates if name in supplied or name in native or name in official), None)
                if found is None:
                    missing_modules.append({"source": path, "line": node.lineno, "module": imported, "names": requested})
                    continue
                if found not in native:
                    add(found)
                raw = supplied.get(found, native.get(found))
                available, wildcard = top_names(raw)
                for name in requested:
                    child = imported.replace(".", "/") + "/" + name + ".py"
                    child_init = imported.replace(".", "/") + "/" + name + "/__init__.py"
                    child_found = next((entry for entry in (child, child_init) if entry in official), None)
                    if name not in available and child_found and child_found not in native:
                        add(child_found)
                    elif name not in available and name != "*" and not wildcard and "__getattr__" not in available:
                        missing_symbols.append({"source": path, "line": node.lineno, "module": imported, "name": name,
                                                "baseline_path": found, "baseline_sha256": sha(raw)})
            if len(supplied) > 320:
                raise ValueError("Dependency closure budget exceeded; inspect rather than copying unrelated whole framework")
        DEST.mkdir(parents=True)
        records, patches = [], []
        for path, raw in sorted(supplied.items()):
            original = official[path]
            blob = hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
            if blob != blobs["python/" + path]["sha"]:
                raise ValueError("Archive/source tree blob differs: " + path)
            target = DEST / "overlay" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            records.append({"path": "overlay/" + path, "bytes": len(raw), "sha256": sha(raw),
                            "upstream_sha256": sha(original), "upstream_git_blob_sha1": blob,
                            "upstream_unchanged": raw == original})
            if raw != original:
                patches.extend(difflib.unified_diff(original.decode().splitlines(True), raw.decode().splitlines(True),
                                                   fromfile="a/" + path, tofile="b/" + path))
        lock = {"schema": "qwen21-das007-overlay-v4", "upstream_commit": COMMIT,
                "upstream_source_archive_sha256": sha(upstream_archive.read_bytes()),
                "native_wheel": wheel.name, "native_wheel_sha256": WHEEL_SHA,
                "native_wheel_python_files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in native.items()},
                "files": records, "port_changes": "Explicit phase-one serial eager model probe: native OffloadableDiTMixin; graph replay cache reads eager row count; VAE spatial parallel branch removed and rejected. Decoder/encoder/attention math retained. Full pipeline orchestration is excluded.",
                "dependency_modules_resolved": len(scanned), "remaining_static_module_gaps": missing_modules,
                "remaining_static_symbol_gaps": missing_symbols, "profile": {"phase": "core-import-and-tiny-cpu", "TP": 1, "SP": 1, "quantization": False, "cuda_graph": False, "compile": False, "layerwise_offload": False, "spatial_parallel_vae": False, "full_service_ready": False, "vision_attention": "upstream Torch SDPA fallback"}, "dynamic_imports_verified": False,
                "GPU_executed": False, "weights_loaded": False, "official_accuracy_verified": False}
        (DEST / "source-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        (DEST / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
        (DEST / "LICENSE-upstream-APACHE-2.0").write_bytes(current.read(prefix + "LICENSE"))
        print(json.dumps({"candidate": str(DEST), "real_modules": len(records), "module_gaps": len(missing_modules),
                          "symbol_gaps": len(missing_symbols), "native_wheel_modified": False}))


if __name__ == "__main__":
    main()
