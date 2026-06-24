#!/usr/bin/env bash
#
# One-shot installer for the AIQ-Contextual-Drag conda environment.
#
# Two steps in a single command:
#   1. conda env create -f env/environment-ica.yml          (heavy: vllm + torch)
#   2. conda run -n phase1-dry-run-eval pip install -e <repo packages>
#
# The second step has to run after env create because conda generates the
# pip requirements file under /tmp and pip resolves `-e <relative_path>`
# against the requirements file's directory, not the user's CWD. Embedding
# the editable lines directly in environment-ica.yml does NOT work.
#
# Usage:
#   bash scripts/install.sh                # default: env=phase1-dry-run-eval, file=env/environment-ica.yml
#   ICA_NEW=1 bash scripts/install.sh      # use env-ica-new.yml + name phase1-dry-run-eval-new
#   ENV_NAME=foo bash scripts/install.sh   # override env name
#
# Run from the AIQ-Contextual-Drag repo root.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [[ "${ICA_NEW:-0}" == "1" ]]; then
    env_file="env/environment-ica-new.yml"
    env_name="${ENV_NAME:-phase1-dry-run-eval-new}"
else
    env_file="env/environment-ica.yml"
    env_name="${ENV_NAME:-phase1-dry-run-eval}"
fi

if [[ ! -f "$env_file" ]]; then
    echo "[install] expected env file at $env_file (run from repo root)" >&2
    exit 1
fi

if [[ ! -d submodules/aiq-magnet ]] || [[ -z "$(ls -A submodules/aiq-magnet 2>/dev/null)" ]]; then
    echo "[install] submodules/aiq-magnet is empty — did you forget --recurse-submodules?" >&2
    echo "[install]   git submodule update --init --recursive" >&2
    exit 1
fi

echo "[install] step 1/3: conda env create ($env_name from $env_file) ..."
conda env create -f "$env_file" -n "$env_name"

echo "[install] step 2/3: pip install -e ./submodules/aiq-magnet ..."
conda run --live-stream -n "$env_name" pip install -e ./submodules/aiq-magnet

echo "[install] step 3/3: pip install -e . ..."
conda run --live-stream -n "$env_name" pip install -e .

echo
echo "[install] done. Activate with:  conda activate $env_name"
echo "[install] smoke test:           PYTHONPATH=$PWD/src python -m magnet.evaluation cards/smoke_runs/Qwen3_8B_NoThinking/wiring/math500.yaml"
