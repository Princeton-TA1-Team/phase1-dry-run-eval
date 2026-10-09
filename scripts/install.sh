#!/usr/bin/env bash
#
# One-shot installer: a conda environment with vLLM, MAGNET 0.1.0 and this
# package, ready to run cards/contextual_drag_ccg_transfer.yaml.
#
#   1. conda env create -f env/environment-ica.yml       (vLLM 0.10.2, torch, evaluation stack)
#   2. pip install -e ".[magnet,analysis,eval]"          (this package and MAGNET 0.1.0)
#
# The editable install runs after `env create` because conda writes the yml's
# pip block to a file under /tmp, and pip resolves `-e <relative_path>`
# against that file's directory rather than the repository.
#
# Usage, from anywhere:
#   bash scripts/install.sh                   # environment phase1-dry-run-eval
#   ENV_NAME=foo bash scripts/install.sh      # another environment name

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

env_name="${ENV_NAME:-phase1-dry-run-eval}"

echo "[install] step 1/2: conda env create ($env_name from env/environment-ica.yml) ..."
conda env create -f env/environment-ica.yml -n "$env_name"

echo "[install] step 2/2: this package and MAGNET 0.1.0 ..."
conda run --live-stream -n "$env_name" pip install -e ".[magnet,analysis,eval]"

echo
echo "[install] done. Activate with:  conda activate $env_name"
echo "[install] check the setup:      bash scripts/run_ccg_transfer.sh --dry_run True"
