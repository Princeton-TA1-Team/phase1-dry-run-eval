#!/usr/bin/env bash
#
# Run the Phase I Final card, cards/contextual_drag_ccg_transfer.yaml.
#
#   bash scripts/run_ccg_transfer.sh                     # full run
#   bash scripts/run_ccg_transfer.sh --dry_run True      # compile the pipeline only
#
# Environment variables:
#   OUTPUT_PATH   where results go (default: evaluation_runs)
#   BACKEND       kwdagger backend: serial (default, runs in the foreground) or tmux
#   CONTEXTUAL_DRAG_ENDPOINT   OpenAI-compatible base URL ending in /v1; unset
#                              means each stage loads the model on the local GPU
#
# Any further arguments are passed to `magnet evaluate_new`.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# The pipeline's stages import `cards` from the repository root and
# `contextual_drag` from src/.
export PYTHONPATH="$repo_root:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python -c "import magnet, sys; sys.exit(0 if magnet.__version__.startswith('0.1') else 1)" 2>/dev/null; then
    echo "[run] MAGNET 0.1.x is not installed in this environment." >&2
    echo "[run] Install it with:  bash scripts/install.sh   (or pip install -e \".[magnet,analysis,eval,inference]\")" >&2
    exit 1
fi

exec python -m magnet evaluate_new cards/contextual_drag_ccg_transfer.yaml \
    --output_path "${OUTPUT_PATH:-evaluation_runs}" \
    --backend "${BACKEND:-serial}" \
    "$@"
