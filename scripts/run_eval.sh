#!/usr/bin/env bash
# =============================================================================
# Clean SEQUENTIAL Contextual-Drag evaluation pipeline for ANY model. The model
# and tensor-parallel size are positional args (default: GPT-OSS-20B, TP=4). No
# scheduler dependency: it holds the local GPU(s) and runs every card one at a time.
#
# Usage (from the repo root):
#   bash scripts/run_eval.sh                          # GPT-OSS-20B, TP=4 (default)
#   bash scripts/run_eval.sh <MODEL_ALIAS> <TP>       # any registered alias + TP
#   bash scripts/run_eval.sh Qwen3_8B_NoThinking 1    # e.g. a dense 8B on 1 GPU
#   SMOKE=1 bash scripts/run_eval.sh                  # reduced-size wiring check
#   RUN_RECURSIVE=1 bash scripts/run_eval.sh          # also run recursive cards (multi-day!)
#
# Under SLURM, wrap it (no committed .slurm needed):
#   sbatch --gres=gpu:4 --time=2-00:00:00 --cpus-per-task=32 --mem=256G \
#          --output=slurm_logs/%x-%j.out --wrap='bash scripts/run_eval.sh'
#
# Phases (strictly sequential):
#   1. render the formal cards (gitignored build artifacts) at the right sizing
#   2. GPU sweep: drag, drag-1f, error-conditioning-external, mitigation, ted
#      (drag-1f produces the shared 1F conditioned-inference cache, runs/cond_cache)
#   3. CPU post-hoc: error-conditioning-posthoc (pure analysis; reuses that cache)
#   4. (optional) recursive_filter1 / recursive_naive cards  (GPT-OSS-20B only)
#
# Overrides: positional MODEL/TP, or env REPO, ENV_NAME, MODEL, TP, RUN_ROOT,
#            SMOKE, RUN_RECURSIVE.
# =============================================================================
set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO" || { echo "FATAL: repo not found: $REPO"; exit 1; }

MODEL="${1:-${MODEL:-GPT_OSS_20B}}"      # positional arg 1, else $MODEL, else default
TP="${2:-${TP:-4}}"                      # positional arg 2, else $TP, else 4 (gpt-oss MoE)
RUN_ROOT="${RUN_ROOT:-runs/run_${MODEL}_$(date +%Y%m%d_%H%M%S)}"
CARDS_DIR="$RUN_ROOT/cards"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$CARDS_DIR" "$LOG_DIR"

echo "=== $MODEL eval pipeline @ $(date) | REPO=$REPO | RUN_ROOT=$RUN_ROOT | TP=$TP ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

# --- environment (portable; override ENV_NAME; default matches scripts/install.sh) ---
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    ENV_NAME="${ENV_NAME:-phase1-dry-run-eval}"
    conda activate "$ENV_NAME" || { echo "FATAL: cannot activate conda env '$ENV_NAME' (set ENV_NAME=...)"; exit 1; }
    echo "[env] conda env: $ENV_NAME"
fi
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"      # resolve contextual_drag to THIS checkout
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== preflight: stack imports ==="
python - <<'PY' || { echo "FATAL: stack import failed (wrong env? set ENV_NAME=...)"; exit 1; }
import vllm, contextual_drag, magnet, scriptconfig, kwdagger
print("vllm", vllm.__version__, "| contextual_drag", contextual_drag.__file__)
PY

# --- 1. render formal cards (build artifacts) at the requested sizing + force TP ---
RENDER_ARGS=(--model "$MODEL" --out_dir "$CARDS_DIR"
             --test drag drag-1f error-conditioning-external error-conditioning-posthoc mitigation ted)
if [[ "${SMOKE:-0}" == 1 ]]; then
    echo "=== SMOKE: reduced sizing (max_questions=32 n=8 max_tokens=8192) ==="
    RENDER_ARGS+=(--max_questions 32 --n 8 --max_tokens 8192)
fi
echo "=== [1/4] rendering formal cards for $MODEL -> $CARDS_DIR ==="
python -m cards.render_formal "${RENDER_ARGS[@]}"
python - "$CARDS_DIR/$MODEL" "$TP" <<'PYTP'
import sys, glob, yaml
class _D(yaml.SafeDumper): pass
_D.add_representer(str, lambda d, x: d.represent_scalar("tag:yaml.org,2002:str", x, style="|" if "\n" in x else None))
root, tp = sys.argv[1], int(sys.argv[2])
for f in glob.glob(root + "/**/*.yaml", recursive=True):
    if "error-conditioning-posthoc" in f:      # analysis card has no GPU params
        continue
    c = yaml.safe_load(open(f)); ap = c["pipeline"][next(iter(c["pipeline"]))]["algo_params"]
    ap["tensor_parallel_size"] = tp
    open(f, "w").write(yaml.dump(c, Dumper=_D, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100))
print(f"[render] tensor_parallel_size={tp} on rendered {root.split('/')[-1]} GPU cards")
PYTP

card_name () { echo "$1" | sed "s#^$CARDS_DIR/##; s#\.yaml\$##; s#/#__#g"; }
SUMMARY="$RUN_ROOT/summary.tsv"; : > "$SUMMARY"

run_card () {  # $1=card  $2=tag  $3=quiet(0/1)
    local card="$1" tag="$2" quiet="${3:-0}" name ec v st
    name="$(card_name "$card")"
    echo; echo ">>> [$tag] START $(date '+%F %T')  $card"
    if [[ "$quiet" == 1 ]]; then
        python -m magnet.evaluation "$card" --output_path "$RUN_ROOT/eval/$name" > "$LOG_DIR/$name.log" 2>&1
        ec=$?
    else
        set -o pipefail
        python -m magnet.evaluation "$card" --output_path "$RUN_ROOT/eval/$name" 2>&1 | tee "$LOG_DIR/$name.log"
        ec=${PIPESTATUS[0]}; set +o pipefail
    fi
    v="$(find "$RUN_ROOT/eval/$name" -name verdict.json 2>/dev/null | head -1)"
    st="(no verdict.json)"
    [[ -n "$v" ]] && st="$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status","?"))' "$v" 2>/dev/null || echo '?')"
    printf '%s\t%s\t%s\n' "$name" "$ec" "$st" >> "$SUMMARY"
    echo "<<< [$tag] DONE  $(date '+%F %T')  exit=$ec  verdict=$st"
}

# --- 2. GPU sweep (sequential) ---
mapfile -t GPU_CARDS < <(ls \
    "$CARDS_DIR/$MODEL"/drag/*.yaml \
    "$CARDS_DIR/$MODEL"/drag-1f/*.yaml \
    "$CARDS_DIR/$MODEL"/error-conditioning-external/*.yaml \
    "$CARDS_DIR/$MODEL"/mitigation/*.yaml \
    "$CARDS_DIR/$MODEL"/ted/*.yaml 2>/dev/null)
N=${#GPU_CARDS[@]}
echo "=== [2/4] GPU sweep: $N cards SEQUENTIALLY (TP=$TP) ==="
i=0; for c in "${GPU_CARDS[@]}"; do i=$((i+1)); run_card "$c" "$i/$N" 0; done

# --- 3. CPU post-hoc analysis (reuses the drag-1f cond_cache; no GPU) ---
echo "=== [3/4] CPU post-hoc: error-conditioning-posthoc (reuses cond_cache) ==="
for c in "$CARDS_DIR/$MODEL"/error-conditioning-posthoc/*.yaml; do
    [[ -f "$c" ]] && run_card "$c" "posthoc" 1
done

# --- 4. recursive (optional; multi-day — better parallelized per-card on a cluster) ---
if [[ "${RUN_RECURSIVE:-0}" == 1 ]]; then
    echo "=== [4/4] recursive cards (committed; n_runs x n_samples_solve x steps; MULTI-DAY) ==="
    for c in cards/formal_test/${MODEL}_recursive/*/*.yaml; do
        [[ -f "$c" ]] || continue
        name="$(echo "$c" | sed 's#cards/formal_test/##; s#\.yaml$##; s#/#__#g')"
        echo ">>> recursive START $(date '+%F %T')  $c"
        python -m magnet.evaluation "$c" --output_path "$RUN_ROOT/recursive/$name" 2>&1 | tee "$LOG_DIR/$name.log"
        echo "<<< recursive DONE  $(date '+%F %T')  $c"
    done
else
    echo "=== [4/4] recursive SKIPPED (set RUN_RECURSIVE=1 to include; multi-day) ==="
fi

echo; echo "######## SUMMARY ($(wc -l < "$SUMMARY") cards) ########"
printf 'CARD\tEXIT\tVERDICT\n' | cat - "$SUMMARY" | { column -t -s "$(printf '\t')" 2>/dev/null || cat; }
echo "Outputs: $RUN_ROOT/eval/   Logs: $LOG_DIR/   Init cache: runs/init_cache/   Cond cache: runs/cond_cache/"
if [ "$MODEL" = "GPT_OSS_20B" ]; then
    echo; echo "=== visualize: python scripts/viz_gpt_oss_20b.py   (auto-picks newest run -> runs/gpt_oss_20b_summary.png) ==="
fi
echo "=== pipeline done @ $(date) ==="
