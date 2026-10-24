#!/usr/bin/env bash
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
