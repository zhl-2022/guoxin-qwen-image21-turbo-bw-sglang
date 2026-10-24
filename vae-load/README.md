# Actual Qwen21 VAE checkpoint CPU probe 010

Run after the009 metadata + VAE phase has14 full-SHA verified files. This probe
strictly binds the sealed009 source bundle manifest and fixed transfer lock,
re-hashes all14 actual files, uses the original complete VAE architecture,
constructs its real SGLang model on meta, and compares every checkpoint key and
shape before `load_state_dict(strict=True, assign=True)`. Missing/unexpected keys
or shapes abort. It then runs32×32 RGBA encode/decode on CPU float32 and records
parameter count, actual dtype/key/shape fingerprints and finite tensor hashes.

```bash
/root/private_data/guoxin-wan-env-006-001/venv/bin/python -B /root/private_data/qwen21-turbo-vae-load010/probe_qwen21_vae_checkpoint010.py --bundle /root/private_data/qwen21-turbo-core-probe009 --output /root/private_data/qwen21-vae-checkpoint010-results --timeout-seconds 300
```

Output must be new and outside source/checkpoint directories. The exact Qwen
checkpoint namespace is read-only. All1,989 native sources are verified before
and after. Imports can initialize a CUDA context; tensors and all compute target
CPU. A Unix alarm requests exit after the bounded timeout; a coordinating parent
should enforce its own process timeout if a native call stops returning.

The original native VAE loader uses strict=False. This dedicated probe instead
rejects every missing/unexpected key or shape and never fills absent parameters.
It does not load text encoder or DiT weights, generate a semantic image, or claim
full-pipeline, GPU, official accuracy, performance, acceptance or submission.
010 is locally Python3.10 syntax checked. Actual native execution is pending.
