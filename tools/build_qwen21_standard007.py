"""Seal fixed Qwen21 code and independently audited actuals into a new directory.

Original archives, model weights, native installations and earlier deliveries are
never rewritten. New engineering tools retain copyright; no global license is
selected for the repository.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile
from audit_qwen21_full014_actual import read_archive
from verify_qwen21_standard007 import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "core": ("session007-qwen21-turbo-core-probe010.zip", "970bf1270fa18d51ab247d9a096c0e2ca6ebb26075cb40b9f3d623592ff0eecb"),
    "pretrained": ("session007-qwen21-pretrained014.zip", "8b52a18dd5a86d187fbacd26fcfc75d94b8073b4385557d0d500d6109e5d523a"),
    "full-shapes": ("session007-qwen21-full-shapes012.zip", "3bb8a32ec31f0af64890da5bad45ee8ccc7eb74383317f6114be3734bc6d99a6"),
    "vae-load": ("session007-qwen21-vae-load010.zip", "84a50ce8d5f8a191be05e9a60529c093355c6b8e11c50320bc6b1291441d963a"),
    "transfer": ("session007-qwen21-vae-transfer009.zip", "c4a5f012340974af10766fd0c6a338ac21c3769ad5a1d1e0d56959e3d438ef48"),
}
ACTUALS = {
    "full014-raw": ("000-session007-qwen21-full014-raw.zip", "d52e54209eafa1328ac7a8b16253da94f6bab368c2b70f14d42d2f2915aa6a8c"),
    "runtime-closure014-raw": ("000-session007-qwen21-runtime-closure014.zip", "38ccf98343e43adaa9ccd58c4ae56003f6ad6784e7efbc22671f05df68cdddd2"),
    "stage010-012-raw": ("000-session007-qwen21-stage010-012-raw.zip", "b6faeea32b4a20eb2cc175df2e477c3d6a47ad5204e974bb4f85bf2344d54dbc"),
}
AUDITS = (
    "qwen21-core010-independent-audit.json", "qwen21-core010-loaded-source-bindings.json",
    "qwen21-stage010-012-independent-audit.json", "qwen21-full014-readiness-gate.json",
    "qwen21-full014-independent-audit.json", "qwen21-full014-contract-audit.json",
    "qwen21-runtime-closure014-independent-audit.json", "qwen21-full014-auditor-selfcheck.json",
    "qwen21-locked-contest-scope-review007.json", "qwen21-decode-order014-real-CPU-regression-b.json",
    "qwen21-standard007-local-validation.json",
)
TOOLS = (
    "audit_qwen21_full014_actual.py", "audit_qwen21_runtime_closure014.py",
    "audit_qwen21_stages013.py", "audit_qwen21_probe_raw007.py",
    "audit_qwen21_loaded_sources007.py", "audit_qwen21_core007.py",
    "audit_qwen21_native_api007.py", "audit_qwen21_gate014.py",
    "qwen21_standard_tools007.py", "verify_qwen21_standard007.py",
    "build_qwen21_standard007.py",
)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def doc(value): return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assert not args.destination.exists() and not args.receipt.exists()
    assert args.output is None or not args.output.exists()
    payload = {}
    identities = {}
    def add(name, raw):
        assert name not in payload and "\\" not in name
        assert not any(part in {".git", "venv", ".venv", "__pycache__", ".cache"} for part in Path(name).parts)
        assert Path(name).suffix.lower() not in {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pyc"}
        payload[name] = raw
    for prefix, (filename, expected) in {**SOURCES, **ACTUALS}.items():
        path = ROOT / "dist" / filename
        raw = path.read_bytes(); assert sha(raw) == expected, filename
        members, count = read_archive(path)
        identities[filename] = {"bytes": len(raw), "sha256": expected, "payloads": count}
        add("archives/" + filename, raw)
        base = prefix + "/" if prefix in SOURCES else "evidence/" + prefix + "/"
        for name, content in members.items(): add(base + name, content)
    strong = json.loads((ROOT / "evidence/session007/qwen21-full014-contract-audit.json").read_text())
    closed = json.loads((ROOT / "evidence/session007/qwen21-runtime-closure014-independent-audit.json").read_text())
    assert strong["full014"][0]["passed"] and len(strong["transfer_receipts"]) == 2 and closed["passed"]
    assert strong["archive_sha256"] == ACTUALS["full014-raw"][1]
    assert closed["archive_sha256"] == ACTUALS["runtime-closure014-raw"][1]
    for name in AUDITS: add("evidence/audits/" + name, (ROOT / "evidence/session007" / name).read_bytes())
    add("evidence/qwen21-full014-contact-sheet.png", (ROOT / "evidence/session007/qwen21-full014-contact-sheet.png").read_bytes())
    for name in ("000-session007-qwen21-core005-raw.zip", "000-session007-qwen21-core007-raw.zip",
            "000-session007-qwen21-core008-raw.zip", "000-session007-qwen21-core009-raw.zip"):
        # Preserve original historical envelopes unchanged. Their separate audits
        # in frozen core/ remain historical results, not new model passes.
        raw = (ROOT / "dist" / name).read_bytes()
        with zipfile.ZipFile(ROOT / "dist" / name) as archive:
            assert archive.testzip() is None
        add("archives/historical/" + name, raw)
    for src, dest in (
        ("evidence/rules/pdf-text.txt", "evidence/official-rules/pdf-text.txt"),
        ("evidence/session004/rules-current-20261023/tracks-current.ax.txt", "evidence/official-rules/tracks-current-20261023.ax.txt"),
        ("evidence/official-materials/20261019/report-template.md", "evidence/official-rules/report-template.md"),
        ("candidates/qwen21-turbo-das007-001/upstream-tree.json", "provenance/upstream-tree.json"),
        ("evidence/session007/qwen21-pipeline-prep009/modelscope-files-master.json", "provenance/modelscope-files-master.json"),
    ): add(dest, (ROOT / src).read_bytes())
    wheels_root = ROOT / "dist/session006-wan-sglang-bundle001"
    wheel_lock = (wheels_root / "wheels-manifest.json").read_bytes()
    add("dependencies/wheels-manifest.json", wheel_lock)
    rows = json.loads(wheel_lock)["wheels"]
    assert len(rows) == 11
    for row in rows:
        raw = (wheels_root / "wheels" / row["filename"]).read_bytes()
        assert len(raw) == row["bytes"] and sha(raw) == row["sha256"]
        add("dependencies/wheels/" + row["filename"], raw)
    add("dependencies/bootstrap_wan_sglang_das_006.py", (ROOT / "scripts/bootstrap_wan_sglang_das_006.py").read_bytes())
    for name in TOOLS: add("tools/" + name, (ROOT / "scripts" / name).read_bytes())
    add("tests/test_standard007.py", (ROOT / "scripts/test_qwen21_standard007.py").read_bytes())
    add("verify_bundle.py", (ROOT / "scripts/verify_qwen21_standard007.py").read_bytes())
    add("README.md", (ROOT / "docs/qwen21-session007-technical-report-final.md").read_bytes())
    add("docs/development-stage014-history.md", (ROOT / "docs/qwen21-full-tp1-stage014.md").read_bytes())
    add("COPYRIGHT-NOTICE.md", "# 版权与许可边界\n\n新增工程工具与适配工作的版权予以保留，具体权利归实际权利人所有。未为整个仓库选择 MIT 或其他新的开源许可。各源文件已有 SPDX、版权声明、Apache-2.0 与第三方 notices 继续有效；Qwen 权重适用保留的 Qwen RESEARCH LICENSE AGREEMENT。离线 wheel 保留其包内原许可。私有备份不代表获得了额外再分发、商业使用或赛事验收授权。\n".encode())
    add(".gitattributes", b"* -text\n")
    add(".gitignore", b"__pycache__/\n*.pyc\n")
    add("env.sh", r'''#!/usr/bin/env bash
export QWEN_TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QWEN_PYTHON="${QWEN_PYTHON:-/root/private_data/guoxin-wan-env-006-001/venv/bin/python}"
export QWEN_OUTPUT="${QWEN_OUTPUT:-/root/private_data/qwen21-standard007-replay001}"
export MODEL_ID="Qwen/Qwen-Image-2.1-Turbo"
export MODEL_REVISION="fc13ba1986fcb905854dc8de2f52eee395f49b5b"
export MODEL_PATH="/root/private_data/guoxin-candidates/models/Qwen-Image-2.1-Turbo-fc13ba1986f"
export FRAMEWORK="native DAS SGLang0.5.12 + fixed Qwen21 source port"
export CARD_TYPE="BW"
export TP=1
export SP=1
export QWEN_RESOLUTIONS="${QWEN_RESOLUTIONS:-256,512}"
export QWEN_MAX_SECONDS="${QWEN_MAX_SECONDS:-1200}"
'''.encode())
    add("run.sh", r'''#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
CMD=("$QWEN_PYTHON" -B "$QWEN_TASK_ROOT/pretrained/run_qwen21_pretrained014.py"
  --bundle "$QWEN_TASK_ROOT/core"
  --expected-bundle-sha256 73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d
  --shape-receipt "$QWEN_TASK_ROOT/evidence/stage010-012-raw/qwen21-full-shapes012-results/full-shapes012.json"
  --resolutions "$QWEN_RESOLUTIONS" --output "$QWEN_OUTPUT" --max-seconds "$QWEN_MAX_SECONDS")
if [[ "${1:-}" == --dry-run && $# -eq 1 ]]; then
  printf 'Plan only; no environment install, download, GPU execution or HTTP service.\n'
  printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi
if [[ $# -ne 0 ]]; then printf 'Only --dry-run is accepted. Configure env.sh variables.\n' >&2; exit 2; fi
"$QWEN_PYTHON" -B "$QWEN_TASK_ROOT/verify_bundle.py" --root "$QWEN_TASK_ROOT"
exec "${CMD[@]}"
'''.encode())
    for name, action in (("env_check.sh", "environment"), ("smoke.sh", "smoke"), ("bench.sh", "bench")):
        suffix = ' --results "$QWEN_OUTPUT"' if action == "smoke" else ""
        raw = '#!/usr/bin/env bash\nset -euo pipefail\nsource "$(dirname -- "$0")/env.sh"\n'
        raw += 'exec "$QWEN_PYTHON" -B "$QWEN_TASK_ROOT/tools/qwen21_standard_tools007.py" ' + action
        raw += ' --root "$QWEN_TASK_ROOT"' + suffix + ' "$@"\n'
        add(name, raw.encode())
    add("verify.py", b'"""Official accuracy material gate; never fabricates a score."""\nfrom pathlib import Path\nimport runpy,sys\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root/"tools"))\nsys.argv=[str(root/"tools/qwen21_standard_tools007.py"),"verify","--root",str(root),*sys.argv[1:]]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n')
    add("task.json", doc({"schema": "qwen21-standard007-development-v1", "model_id": "Qwen/Qwen-Image-2.1-Turbo",
        "revision": "fc13ba1986fcb905854dc8de2f52eee395f49b5b", "card": "BW", "TP": 1, "SP": 1,
        "actual_full_pretrained_GPU_verified": True, "actual_resolutions": [[256, 256], [512, 512]],
        "profile": "T2I same prompt seed42, eight published sigmas, batch1, BF16, eager, real Torch SDPA",
        "standard_wrapper_GPU_replayed": False, "HTTP_service_tested": False, "image_editing_tested": False,
        "resolution1024_tested": False, "official_accuracy_verified": False, "official_performance_verified": False,
        "official_acceptance": False, "submission_ready": False, "submitted": False}))
    add("provenance.json", doc({"schema": "qwen21-standard007-provenance-v1", "frozen_inputs": identities,
        "all_selected_frozen_payload_bytes_preserved": True, "native_wheel_and_previous_venv_modified": False,
        "source_and_checkpoint_bindings": "core/source-lock.json and transfer/download-sources009.json",
        "actual_full014_source_binding": "pretrained/MANIFEST.json and evidence/full014-raw/qwen21-complete-pretrained014-results/complete-pretrained014.json",
        "archive_and_Git_receipts": "External to the sealed tree to avoid self-reference",
        "scope": "Actual pretrained development evidence, not official evaluation or submission"}))
    manifest = doc({"schema": "qwen21-standard007-exact-delivery-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "official_accuracy_verified": False, "official_performance_verified": False, "submitted": False,
        "files": [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(payload.items())]})
    payload["delivery-manifest.json"] = manifest
    target = args.destination.absolute()
    assert not any(p.is_symlink() for p in (target, *target.parents))
    target.mkdir(parents=True)
    for name, raw in payload.items():
        path = target / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream: stream.write(raw)
    integrity = verify(target, sha(manifest))
    archive_identity = None
    if args.output:
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in sorted(payload.items()):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                archive.writestr(info, raw)
        with zipfile.ZipFile(args.output) as archive:
            assert archive.testzip() is None and set(archive.namelist()) == set(payload)
            for name, raw in payload.items(): assert archive.read(name) == raw
        archive_identity = {"path": str(args.output.absolute()), "bytes": args.output.stat().st_size, "sha256": sha(args.output.read_bytes())}
    receipt = {"schema": "qwen21-standard007-package-receipt-v1", "target": str(target),
        "archive": archive_identity, "manifest_sha256": sha(manifest), "payload_files": len(payload),
        "integrity": integrity, "all_frozen_code_and_actuals_preserved": True,
        "GPU_execution_in_this_packaging_step": False, "official_acceptance": False, "submission_ready": False, "submitted": False}
    with args.receipt.open("x", encoding="utf-8") as stream: json.dump(receipt, stream, indent=2); stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
