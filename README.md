# AIQ-Contextual-Drag

[AIQ-magnet](https://github.com/AIQ-Kitware/aiq-magnet) integration for *Contextual Drag* (Cheng et al., 2026, [arXiv:2602.04288](https://arxiv.org/abs/2602.04288)). A self-contained, pip-installable package that reproduces the Contextual Drag evaluation suite — six per-`(model, test, dataset)` tests plus recursive self-improvement — rendered from a central registry (`cards/render_formal.py`) and run as AIQ-magnet cards. Each card subprocesses the `contextual-drag` CLI and emits a claim verdict (VERIFIED / FALSIFIED / INCONCLUSIVE).

## Tests

| Test | Node | Claim (threshold) |
|---|---|---|
| `drag` | `run_drag_experiment` | `drag = acc_clean − acc_2f ≥ 0.05` (2F self-drag) |
| `drag-1f` | `run_drag_1f` | `drag_1f = acc_direct − acc_1f ≥ 0.05` (1F; produces the shared 1F cache) |
| `error-conditioning-external` | `run_error_conditioning` (framing) | `delta_acc ≥ 0.05` (1F, prompt states draft is wrong; no verdict filter) |
| `error-conditioning-posthoc` | `run_ec_posthoc` (analysis, no GPU) | `delta_acc ≥ 0.05` (verdict-filtered; reuses the drag-1f cache) |
| `mitigation` | `run_mitigation_experiment` | `recovery_rate ≥ 0.20` (`cm_filter1`) |
| `ted` | `run_ted_experiment` | `ted_drag ≥ 1.0` (24-game only) |
| `recursive_filter1` / `recursive_naive` | `run_recursive_{filter1,naive}` | `delta_acc_rf1 ≥ +0.02` / `delta_acc_naive ≤ −0.05` (GPT-OSS-20B only) |

**Datasets** (8): `aime24 aime25 hmmt24 hmmt25 gpqa mmlu crux-i 24-game`. `ted` is 24-game only; recursive uses the 4 math sets (`aime24 aime25 hmmt24 hmmt25`).

**Cards.** Small wiring/claim checks on bundled `data/smoke/` slices live at `cards/smoke_runs/<model>/<test>/<dataset>.yaml`. Full-dataset "formal" cards at `cards/formal_test/<model>/<test>/<dataset>.yaml` are rendered build artifacts (`python -m cards.render_formal`, gitignored) over `data/full_data/`. Thresholds are card algo_params (sweepable via magnet).

## Quickstart

```bash
git clone --recurse-submodules <url> && cd AIQ-Contextual-Drag
bash scripts/install.sh                 # conda env + editable installs (aiq-magnet + contextual_drag)
conda activate phase1-dry-run-eval

# (optional) 1-GPU plumbing check that the install works end-to-end:
python -m magnet.evaluation cards/smoke_runs/Qwen3_8B_NoThinking/wiring/math500.yaml

# GPT-OSS-20B study as ONE sequential, scheduler-free script (needs 4x80GB, TP=4):
SMOKE=1 bash scripts/run_eval.sh   # quick reduced-size wiring check
bash scripts/run_eval.sh           # full evaluation
python scripts/viz_gpt_oss_20b.py         # -> runs/gpt_oss_20b_summary.png (auto-picks newest run)
```

## Running the full experiment set

### GPT-OSS-20B

One sequential, scheduler-free script does the whole study — renders the formal cards, runs `drag / drag-1f / error-conditioning-external / mitigation / ted` one at a time on the local GPU(s), then `error-conditioning-posthoc` as a CPU post-hoc pass, then prints the visualization command:

```bash
bash scripts/run_eval.sh                  # full non-recursive suite
RUN_RECURSIVE=1 bash scripts/run_eval.sh  # also run the recursive cards (multi-day)
python scripts/viz_gpt_oss_20b.py                # summary plot (recursive panel reads runs/recursive_full_*)
```

On a cluster, wrap it (cluster `.slurm` wrappers are gitignored / local-only):

```bash
sbatch --gres=gpu:4 --time=2-00:00:00 --cpus-per-task=32 --mem=256G \
       --partition=<your_partition> --output=slurm_logs/%x-%j.out \
       --wrap='bash scripts/run_eval.sh'
```

### Other models

The renderer and nodes are model-agnostic (dataset-family dispatch, every verifier, and `<think>` / no-thinking / gpt-oss-harmony thinking parsing are centralized), so a new model is mechanical:

1. **Register the alias** in `src/contextual_drag/resources/inference/eval_models_params.json` (`model_name`, `context_length`, `sampling_params`); cache its weights locally (compute nodes run offline, `HF_HUB_OFFLINE=1`).
2. **Launch** — the same script is `MODEL`/`TP`-parameterized:

```bash
bash scripts/run_eval.sh Qwen3_8B_NoThinking 1
```

Pick `TP` by model size: dense ≤ ~13B → `TP=1` (1 GPU); ~20–32B / large MoE → `TP=4` (4 GPUs); 70B+ → `TP=4–8`. Recursive is GPT-OSS-20B-only.

## Experiment configuration

- **Sizing.** Formal-card defaults: full dataset, `n=16`, per-model `max_tokens` (from `eval_models_params.json`, paper Table 2). Override at render time: `python -m cards.render_formal --model <alias> [--test ...] [--n N] [--max_tokens T] [--max_questions Q]`. `run_eval.sh` forces `tensor_parallel_size=TP` on the rendered GPU cards; `SMOKE=1` renders reduced sizing (`max_questions=32 n=8 max_tokens=8192`).
- **Shared init-sampling cache.** drag / drag-1f / error-conditioning-external / mitigation / ted reuse one clean init pass per `(model, dataset)` under `runs/init_cache/<model>/<dataset>/` (`--init_cache_root`, default `runs/init_cache`; `''` to disable). Reused on a matching init config (model/dataset/template/`n`/`max_tokens`/`max_questions`), else a card-local fallback (never stale).
- **Shared 1F conditioned-inference cache.** `drag-1f` computes the 1F conditioned inference + analysis once and writes `ec_summary.json` (raw + verdict-filtered metrics) to `runs/cond_cache/<model>/<dataset>/1f/` (`--cond_cache_root`). `error-conditioning-posthoc` is pure analysis — it reads that cache (no GPU) and reports the verdict-filtered metric, so `drag-1f` must run first (the pipeline script orders this; on a cold cache the posthoc card returns `cache_missing: true`).
- **Recursive.** `n_runs=16` independent trajectories × `n_samples_solve=16` rollouts/step × `max_recursive_steps=16`; per-step pass@1 is averaged over `n_runs × n_samples_solve` generations (std across trajectories; see `cards/nodes/_recursive_multirun.py`). The card auto-runs Stage -1 init-sampling when pointed at a raw `data/full_data/<task>/<task>.ds`.
- **Data.** `data/smoke/<task>/<task>.ds` are bounded wiring fixtures; `data/full_data/<task>/<task>.ds` are the full benchmarks. Inference/eval stream JSONL keyed by `prompt_hash`, so runs are kill-safe and resume per-row.

Every CLI verb prints its flags under `contextual-drag <verb> --help`; see `cards/nodes/run_*.py` for the subprocess chain each card invokes.

## Environment

`scripts/install.sh` creates the conda env and editable-installs `submodules/aiq-magnet` + the package. All runs need `PYTHONPATH=$PWD/src` (the pipeline script sets it). `run_eval.sh` defaults to env `phase1-dry-run-eval`; override with `ENV_NAME=...`.

| Env | vLLM | Use for |
|---|---|---|
| `phase1-dry-run-eval` (`env/environment-ica.yml`) | 0.10.2 | GPT-OSS, Nemotron, Qwen3, R1-Distill, Llama3.1, SFT, GRPO |
| `phase1-dry-run-eval-new` (`env/environment-ica-new.yml`) | 0.19.1 | newer families (`ICA_NEW=1 bash scripts/install.sh`) |

## Unit tests

```bash
pip install -e .[dev,eval,analysis,magnet]      # vllm is stubbed in conftest
PYTHONPATH=$PWD/src pytest tests/ -q
```

## Citation

```bibtex
@article{cheng2026contextual,
  title        = {Contextual Drag: How Errors in the Context Affect LLM Reasoning},
  author       = {Cheng et al.},
  year         = {2026},
  journal      = {arXiv preprint},
  eprint       = {2602.04288},
  archivePrefix= {arXiv}
}
```

Produced jointly by Princeton-PLI and Kitware as an AIQ-magnet integration for the Contextual Drag evaluation.
