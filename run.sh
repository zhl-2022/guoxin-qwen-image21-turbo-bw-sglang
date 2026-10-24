#!/usr/bin/env bash
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
