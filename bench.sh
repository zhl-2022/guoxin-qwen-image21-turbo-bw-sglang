#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
exec "$QWEN_PYTHON" -B "$QWEN_TASK_ROOT/tools/qwen21_standard_tools007.py" bench --root "$QWEN_TASK_ROOT" "$@"
