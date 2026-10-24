# Qwen-Image 2.1 Turbo metadata and VAE transfer 009

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
