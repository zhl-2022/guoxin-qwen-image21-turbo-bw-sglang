"""Fix two actual009 native interfaces in a new frozen core010 package."""
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
    dest = ROOT / "candidates/qwen21-turbo-das007-010"
    assert not dest.exists()
    shutil.copytree(ROOT / "candidates/qwen21-turbo-das007-009", dest)
    (dest / "MANIFEST.json").unlink()
    lock = json.loads((dest / "source-lock.json").read_text())
    dit = dest / "overlay/sglang/multimodal_gen/runtime/models/dits/qwen_image21.py"
    text = dit.read_text()
    imported = "from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention"
    assert text.count(imported) == 1
    text = text.replace(imported, "from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention as NativeUSPAttention")
    adapter = '''class USPAttention(NativeUSPAttention):
    """Exact upstream SP1 replicated-prefix branch on genuine native attention."""

    def forward_with_replicated_kv_prefix(self, q, k_prefix, v_prefix, k_suffix, v_suffix):
        if get_sp_world_size() != 1:
            raise ValueError("Qwen21 DAS profile requires SP1 for replicated KV")
        forward_context = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        k = torch.cat([k_prefix, k_suffix], dim=1)
        v = torch.cat([v_prefix, v_suffix], dim=1)
        return self.attn_impl.forward(q, k, v, ctx_attn_metadata)


'''
    assert text.count("_is_cuda = current_platform.is_cuda()") == 1
    text = text.replace("_is_cuda = current_platform.is_cuda()", adapter + "_is_cuda = current_platform.is_cuda()")
    dit.write_text(text, encoding="utf-8")
    probe = dest / "probe_qwen21_core007.py"
    text = probe.read_text()
    function = '''def init_native_srt_args():
    # Real SRT ServerArgs, in its documented no-serving initialization mode.
    # Qwen weights/model configs are separately bound by the MM execution path.
    from sglang.srt.server_args import (
        ServerArgs as SRTServerArgs, set_global_server_args_for_scheduler,
        get_global_server_args)
    args = SRTServerArgs(model_path="None", device="cpu", rl_on_policy_target=None)
    set_global_server_args_for_scheduler(args)
    assert get_global_server_args() is args and args.rl_on_policy_target is None
    return {"class": type(args).__module__ + "." + type(args).__name__,
            "purpose": "Auxiliary native RoPE configuration; no SRT model server",
            "model_path": args.model_path, "device": args.device,
            "rl_on_policy_target": args.rl_on_policy_target}


'''
    assert text.count("def run_component(name, root, output):") == 1
    text = text.replace("def run_component(name, root, output):", function + "def run_component(name, root, output):")
    assert text.count("    set_global_server_args(args)\n") == 1
    text = text.replace("    set_global_server_args(args)\n", "    set_global_server_args(args)\n    init_native_srt_args()\n")
    probe.write_text(text, encoding="utf-8")
    for name in ("qwen21_t2i_stage011.py", "probe_qwen21_pipeline011.py"):
        raw = (ROOT / "scripts" / name).read_bytes()
        ast.parse(raw.decode(), feature_version=(3, 10))
        (dest / name).write_bytes(raw)
    shutil.copyfile(ROOT / "evidence/session007/qwen21-stage011-local-real-math-probe-b.json", dest / "stage011-local-real-math-probe.json")
    for name in ("qwen21-core009-independent-audit.json", "qwen21-core009-loaded-source-bindings.json"):
        shutil.copyfile(ROOT / "evidence/session007" / name, dest / name)
    source_path = ROOT / "dist/sglang-qwen21-efc984-source007.zip"
    assert sha(source_path.read_bytes()) == lock["upstream_source_archive_sha256"]
    tree = json.loads((ROOT / "candidates/qwen21-turbo-das007-001/upstream-tree.json").read_text())
    blobs = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    records, patches = [], []
    with zipfile.ZipFile(source_path) as archive:
        assert archive.testzip() is None
        prefix = archive.namelist()[0] + "python/"
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
                "upstream_sha256": sha(original), "upstream_git_blob_sha1": blob, "upstream_unchanged": raw == original})
            if raw != original:
                patches.extend(difflib.unified_diff(original.decode().splitlines(True), raw.decode().splitlines(True),
                    fromfile="a/" + relative, tofile="b/" + relative))
        s = archive.read(prefix + "sglang/multimodal_gen/runtime/layers/attention/layer.py").decode()
        module = ast.parse(s)
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "USPAttention")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_with_replicated_kv_prefix")
        original_branch = next(node for node in method.body if isinstance(node, ast.If)).body
        current = ast.parse(dit.read_text())
        cls = next(node for node in current.body if isinstance(node, ast.ClassDef) and node.name == "USPAttention")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef))
        current_branch = method.body[-3:]
        assert [ast.dump(node, include_attributes=False) for node in original_branch] == [ast.dump(node, include_attributes=False) for node in current_branch]
        (dest / "usp-SP1-branch-closure010.json").write_text(json.dumps({"native_subclass": True,
            "upstream_SP1_attention_branch_AST_identical": True, "SP_above1_rejected": True,
            "native_attention_impl_preserved": True, "new_NVIDIA_JIT_used": False}, indent=2) + "\n", encoding="utf-8")
    lock["files"] = records
    lock["actual009_raw_sha256"] = "1fdb4afe28a17d0c3df0be6e34da9217dad9837ddb3dcf3f998fb934e043a3b2"
    lock["actual009_raw_independently_audited"] = True
    lock["port_changes"].extend([
        "010: concrete model-local native USPAttention subclass adds exactly upstream SP1 KV concatenation + real attn_impl.forward; SP>1 rejected, no new JIT branch.",
        "010: initialize real SRT ServerArgs in supported no-serving mode via real setter before native RoPE; separate from genuine MM ServerArgs and actual Qwen model binding."])
    (dest / "source-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (dest / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
    shutil.copyfile(Path(__file__), dest / "build010-source.py")
    (dest / "README.md").write_text('''# Native Qwen21 core010 and executable five-component chain011

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
''', encoding="utf-8")
    files = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            raw = path.read_bytes(); files.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-probe-bundle-v1", "files": files}, indent=2) + "\n", encoding="utf-8")
    manifest_sha = sha((dest / "MANIFEST.json").read_bytes())
    archive_path = ROOT / "dist/session007-qwen21-turbo-core-probe010.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file(): archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive_path) as archive: assert archive.testzip() is None
    value = sha(archive_path.read_bytes())
    helper = ROOT / "scripts/unpack_qwen21_core_probe007_010.py"
    helper.write_text((ROOT / "scripts/unpack_qwen21_core_probe007_009.py").read_text().replace(
        "c918dd034d6e807deab79a10dd9599871aee0f0370ebdda1ccba742877c2ca50", value), encoding="utf-8")
    result = {"zip": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": value,
        "manifest_sha256": manifest_sha, "helper": str(helper), "helper_sha256": sha(helper.read_bytes()),
        "payloads": len(files), "overlay_source_files": len(records), "dynamic010_verified": False, "GPU_executed": False}
    with (ROOT / "evidence/session007/qwen21-core010-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
