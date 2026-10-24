"""Seal a stdlib-only metadata/VAE phase transfer, with an independent namespace."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    dest = ROOT / "tasks/qwen21-turbo-stage009-transfer"
    assert not dest.exists()
    dest.mkdir()
    for name in ("download_qwen21_turbo009.py", "download_candidate_models_20261023.py", "download_wan_models_006_002.py", "download_wan_models_006.py"):
        raw = (ROOT / "scripts" / name).read_bytes()
        ast.parse(raw.decode(), feature_version=(3, 10))
        (dest / name).write_bytes(raw)
    lock = ROOT / "evidence/session007/qwen21-pipeline-prep009/download-sources009.json"
    assert sha(lock.read_bytes()) == "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"
    shutil.copyfile(lock, dest / lock.name)
    shutil.copyfile(ROOT / "evidence/session007/qwen21-pipeline-prep009/modelscope-files-master.json", dest / "modelscope-files-observation.json")
    shutil.copyfile(ROOT / "candidates/qwen21-turbo-das007-008/model-assets/LICENSE", dest / "LICENSE-model-Qwen-RESEARCH")
    tests = []
    for name in ("test_qwen21_transfer009.py", "test_download_wan_parallel_006_002.py"):
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", name, "-v"],
            cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        tests.append({"test_file": name, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        shutil.copyfile(ROOT / "tests" / name, dest / name)
    (dest / "local-validation.json").write_text(json.dumps({"suites": tests,
        "tests_passed": 12, "tests_skipped": 1, "skip": "Actual Linux SIGINT test cannot run on Windows",
        "actual_checkpoint_download_executed": False, "GPU_imported": False}, indent=2) + "\n", encoding="utf-8")
    (dest / "README.md").write_text('''# Qwen-Image 2.1 Turbo metadata and VAE transfer 009

This stdlib-only package downloads a bounded selected phase into a separate Qwen
checkpoint directory. It neither installs Python dependencies nor imports Torch.
The default1GiB byte budget permits metadata + VAE and rejects the complete
32.45GB checkpoint phase. Full-model download remains a separate execution step.

## Fixed identities

Model: Qwen/Qwen-Image-2.1-Turbo. Hugging Face revision:
`fc13ba1986fcb905854dc8de2f52eee395f49b5b`.
All17 file lengths and SHA256 values match this revision and the observed public
ModelScope repository. Each transfer URL uses that file's immutable40-digit
ModelScope commit. The complete source lock SHA256 is
`dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd`.

| Phase | Files | Exact bytes |
|---|---:|---:|
| metadata | 13 | 11,488,566 |
| vae (including metadata) | 14 | 686,997,222 |
| all (explicit larger budget required) | 17 | 32,451,622,214 |

VAE checkpoint alone:675,508,656 bytes, SHA256
`71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c`.

## Execute on native Linux

```bash
python3 -B download_qwen21_turbo009.py plan --component vae
python3 -B download_qwen21_turbo009.py download --component vae --max-seconds 900
python3 -B download_qwen21_turbo009.py verify --component vae --max-seconds 180
```

The destination is
`/root/private_data/guoxin-candidates/models/Qwen-Image-2.1-Turbo-fc13ba1986f`.
Only `qwen21-turbo009-transfer.lock` is acquired with exclusive creation. Existing
locks are never taken over. Receipts are new `qwen21-turbo009-transfer-UUID.json`
files beside the checkpoint directory, and bind this phase to full disk SHA256.
Two transfer workers are joined before the owner lock is released. A shared
deadline stops reads and hashes; owned `.part` plus `.part.json` survive deadline,
failure or interruption and are resumed only when exact length/SHA identity agrees.
Complete files are reused only after their disk SHA passes; no final is overwritten.
The reused reviewed worker functions do not call the Wan controller, namespace
or lock. This Qwen controller never changes those shared module constants.

## Local verification and evidence limits

Five Qwen controller ownership/selection/budget checks and seven shared worker
Range, SHA, cancellation and failure checks passed. One actual Linux SIGINT check
is skipped on this Windows host. Tests use small fixture bytes, no model download.
The observed metadata is source identity evidence, not actual checkpoint bytes.
Only the native download/verify receipt can establish a completed14-file phase.
It cannot establish full32GB loading, image generation, official evaluation,
performance, acceptance or submission. The original Qwen RESEARCH license is
included; no repository-wide relicensing is implied.
''', encoding="utf-8")
    shutil.copyfile(Path(__file__), dest / "build009-source.py")
    files = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            files.append({"path": path.relative_to(dest).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    (dest / "MANIFEST.json").write_text(json.dumps({"schema": "qwen21-transfer-package-v1", "files": files}, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "dist/session007-qwen21-vae-transfer009.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    value = sha(archive_path.read_bytes())
    helper = ROOT / "scripts/unpack_qwen21_vae_transfer009.py"
    text = (ROOT / "scripts/unpack_qwen21_core_probe007_009.py").read_text().replace(
        "c918dd034d6e807deab79a10dd9599871aee0f0370ebdda1ccba742877c2ca50", value)
    helper.write_text(text, encoding="utf-8")
    result = {"zip": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": value,
        "helper": str(helper), "helper_sha256": sha(helper.read_bytes()), "payload_files": len(files),
        "actual_transfer_executed": False, "GPU_imported": False}
    with (ROOT / "evidence/session007/qwen21-transfer009-package-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
