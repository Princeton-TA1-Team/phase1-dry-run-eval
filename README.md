# AIQ-Contextual-Drag

[AIQ-magnet](https://github.com/AIQ-Kitware/aiq-magnet) integration for *Contextual Drag* (Cheng et al., 2026, arXiv:2602.04288): a self-contained, pip-installable Python package that reproduces five evaluation cards on a single GPU.

The package wraps an async vLLM inference driver, a resumable per-row evaluator, a context-manipulation mitigation pipeline, and three analysis modules (error-conditioning, tree-edit distance, mitigation outcome buckets) behind a unified `scriptconfig.ModalCLI`. Five magnet cards subprocess the CLI to produce claim verdicts:

| Card | Claim | Runtime |
|---|---|---|
| `contextual_drag_smoke` | `accuracy ≥ 0.25` on math500 (wiring smoke) | ≤ 3 min |
| `contextual_drag` | `drag = acc_clean − acc_2f ≥ 0.05` (§2 baseline) | ≤ 8 min |
| `contextual_drag_error_conditioning` | `delta_acc = acc_direct − acc_conditioned ≥ 0.05` (§3) | ≤ 12 min |
| `contextual_drag_mitigation` | `recovery_rate ≥ 0.20` for `cm_filter1` (§4) | ≤ 15 min |
| `contextual_drag_ted` | `ted_drag = mean_TED(direct) − mean_TED(2f) ≥ 1.0` | ≤ 10 min |

## Design principles

- **No scheduler dependency.** The host process is assumed to hold its GPU(s) for the duration of a card; there is no Slurm, no SBATCH, no `--dependency=afterany`. All scheduling is the caller's responsibility.
- **Kill-safe checkpointing.** Inference and evaluation stream their outputs as JSONL with append + flush + fsync per row, keyed by `prompt_hash = sha256(rendered_prompt)[:16]`. SIGTERM mid-run drains in-flight requests and exits cleanly; the next invocation resumes at the next unfinished row. Per-stage `.ds` artefact-presence resume covers the mitigation chain.
- **Self-contained.** Every artefact the cards consume is either bundled (smoke datasets in `data/smoke/`, prompt templates in `prompt_templates/`, packaged resources in `src/contextual_drag/resources/`) or passed in by the caller via explicit CLI flags. The package never resolves paths in collaborator scratch directories.
- **Per-cell CLI surface.** Every verb takes one (model, dataset, output\_dir) cell. The multi-cell sweep machinery and external path registry from the upstream research repo are deliberately omitted.

## Quickstart

```bash
git clone --recurse-submodules <url> && cd AIQ-Contextual-Drag
bash scripts/install.sh                          # one-shot: conda env + editable installs of aiq-magnet + contextual_drag
conda activate phase1-dry-run-eval
python -m magnet.evaluation cards/contextual_drag_smoke.yaml
```

`scripts/install.sh` runs `conda env create -f env/environment-ica.yml` and then installs `submodules/aiq-magnet` and the local package via `conda run -n phase1-dry-run-eval pip install -e ...`. The two-step split is intentional: conda dumps the env-yml pip block to `/tmp/condaenv.XXX.requirements.txt` and pip resolves editable paths relative to that file's directory, not your CWD, so embedding `--editable ./submodules/aiq-magnet` directly in `environment-ica.yml` silently breaks. Use `ICA_NEW=1 bash scripts/install.sh` for the vLLM-0.19 environment.

Use `env/environment-ica-new.yml` (and `conda activate phase1-dry-run-eval-new`) for the model families that need vLLM 0.19.

## Running the cards

Each command is run from the repo root on a host with one 24 GB GPU already allocated. Symbol thresholds and target values reflect the Qwen3_8B_NoThinking defaults baked into each card; algo params (model, task, sample count, threshold) are sweepable via magnet.

The "last verified" numbers below come from a clean smoke run on an H100 (80 GB) with `Qwen3_8B_NoThinking`, `max_tokens=8192`, `context_length=32768`, and the card's stock default algo params.

### Wiring smoke

```bash
python -m magnet.evaluation cards/contextual_drag_smoke.yaml
```

- PASS: `accuracy ≥ min_accuracy` (default 0.25; typical ≥ 0.75 for Qwen3_8B_NoThinking on math500).
- Purpose: prove the magnet ↔ contextual_drag plumbing end-to-end. Not a scientific claim.
- **Last verified**: `VERIFIED`, accuracy = 0.5 on 4 math500 problems × n=1.

### §2 baseline drag

```bash
python -m magnet.evaluation cards/contextual_drag.yaml
```

- PASS: `drag ≥ drag_threshold` (default 0.05). Typical measured drag on gpqa: +0.10 to +0.45 depending on which problems land in the aggregate-filtered cohort.
- INCONCLUSIVE: `aggregate_failed: true` — the `≥ num_false` failed-trajectory filter produced no problems. Increase `n` or pick a model whose pass@k on the chosen task sits in the ambiguous zone.
- **Last verified**: `VERIFIED`, drag = +0.094 (acc_clean = 0.531, acc_2f = 0.438), n_kept = 4 on gpqa × 8 problems × n=8.

### §3 error-conditioning

```bash
python -m magnet.evaluation cards/contextual_drag_error_conditioning.yaml
```

- PASS: `delta_acc ≥ delta_threshold` (default 0.05). Target on aime24 × Qwen3_8B × regime `2f` at full sample sizes: ≈ +0.22.
- INCONCLUSIVE: `aggregate_failed: true` or `filter_dropped_all: true` — `data aggregate` returned 0 problems, or the regime-specific verdict filter (`<overall_verdict>incorrect</overall_verdict>` for 1f/2f) rejected every conditioned response. Switch `regime` to `framing` (no verdict filter), or increase `n` / `max_questions`.
- **Last verified**: pipeline-functional end-to-end, but `FALSIFIED` on the stock smoke slice — delta_acc = 0.0 (acc_direct = 0.5, acc_conditioned = 0.5) with only n_kept = 2 problems surviving the 2F aggregate + verdict filter on a 16-problem × n=4 aime24 slice. The pipeline correctly refuses to verify when the kept cohort is this thin; bump `n` to 8 or use the framing regime to drive the claim past threshold.

### §4 mitigation

```bash
python -m magnet.evaluation cards/contextual_drag_mitigation.yaml
```

- PASS: `recovery_rate ≥ recovery_threshold` (default 0.20). Target on gpqa × Qwen3_8B × `cm_filter1`: ≈ 0.40.
- INCONCLUSIVE: `drag_failed_den == 0` — the (Direct ✓, 1F ✗) denominator was empty. Bump `max_questions` or choose a (model, task) cell with stronger measured drag.

### TED structural drag

```bash
python -m magnet.evaluation cards/contextual_drag_ted.yaml
```

- PASS: `ted_drag ≥ ted_threshold` (default 1.0). Target on 24-game × Qwen3_8B × phase `2f`: ≈ 1.5–2.0.
- INCONCLUSIVE: `n_kept_problems == 0` — no problem had a parseable boxed expression in both the anchored and the init responses. Increase `n` or relax the answer parser.

## Repository layout

```
AIQ-Contextual-Drag/
├── pyproject.toml                 # extras: [inference, eval, analysis, magnet, dev, all]
├── env/
│   ├── environment-ica.yml        # vllm 0.10.2 / torch 2.8.0  (old + sft + rl families)
│   └── environment-ica-new.yml    # vllm 0.19.1                (newer model families)
├── src/contextual_drag/
│   ├── cli.py                     # top-level scriptconfig.ModalCLI
│   ├── inference/                 # async vLLM driver, per-cell, prompt-hash JSONL resume
│   ├── evaluation/{math,crux}/    # per-cell resumable evaluator
│   ├── data/                      # aggregation / flatten / postprocess CLIs
│   ├── mitigation/                # cm_filter1 / cm_revise1 pipeline
│   ├── analysis/                  # error_conditioning, ted, mitigation_buckets
│   ├── config/{paths,resources}.py
│   └── resources/                 # eval_models_params.json, cruxeval.jsonl, encodings
├── prompt_templates/              # init, 1f, 2f, framing, cm, error-signal JSON templates
├── cards/
│   ├── contextual_drag_smoke.yaml
│   ├── contextual_drag.yaml
│   ├── contextual_drag_error_conditioning.yaml
│   ├── contextual_drag_mitigation.yaml
│   ├── contextual_drag_ted.yaml
│   └── nodes/                     # one subprocess wrapper per card
├── data/smoke/                    # bundled raw .ds slices + PROVENANCE per task + MANIFEST
│   ├── math500/    (4 rows)
│   ├── gpqa/       (16 rows)
│   ├── aime24/     (16 rows)
│   ├── 24-game/    (32 rows)
│   └── prebuilt/                  # placeholders; cards currently run inline
├── submodules/aiq-magnet/         # git submodule, pinned to a known-good commit
├── tests/                         # pytest suite (CPU-only; vllm stubbed in conftest)
└── .github/workflows/test.yml     # pytest + ruff on push/PR
```

## CLI reference

```
contextual-drag
├── inference      run | dry-run | list-models
├── eval           math | crux | game_of_24
├── data           initial-sampling-postprocess | minimal-aggregate-flatten |
│                  aggregate | aggregate-crux | aggregate-iterative |
│                  stage1-postprocess-iterative
├── mitigation     run                         # --variant cm_filter1 | cm_revise1
└── analysis
    ├── error-conditioning  run | visualize    # local jsonl in, summary json out
    ├── ted                 build-cache | summarize | render
    └── mitigation-buckets  run | render
```

Every verb prints its flag set under `--help`. See `cards/nodes/run_*.py` for the canonical 6–8-step subprocess chain each card invokes.

## Development

### Tests

```bash
pip install -e .[dev,eval,analysis,magnet]   # no [inference]; vllm is stubbed in conftest
pytest tests/ -v
```

Coverage:

- `test_smoke.py` — package import, top-level `--help`, packaged resources load.
- `test_cli.py` — `--help` returns 0 for every CLI verb (parameterised over the full tree).
- `test_cards.py` — every card YAML parses, claim compiles, claim symbols are declared, wrapper `--help` exits 0, subprocess chain wires through the package CLI.
- `test_resume.py` — `load_completed_hashes` skips existing rows, tolerates truncated trailing lines, produces no duplicate hashes after appended rows.
- `test_packaged_resources.py` — `eval_models_params.json`, `cruxeval.jsonl`, tiktoken encodings load via `importlib.resources`.
- `test_prompt_budget.py` — regression catch-net for the `max_tokens − generation_tokens` prompt-truncation bug.
- `test_record_schema.py` — pins the `*_generations_metadata` JSONL record schema.
- `test_aggregate_empty.py` — `data aggregate` exits 1 with a configured-filter message on empty filter result.

### Lint

```bash
ruff check src/contextual_drag/ cards/
```

### Environment management

Two conda environments cover all model families:

| Env | vLLM | Use for |
|---|---|---|
| `phase1-dry-run-eval` (`env/environment-ica.yml`) | 0.10.2 | GPT-OSS, Nemotron, Qwen3, R1-Distill, Llama3.1, SFT, GRPO |
| `phase1-dry-run-eval-new` (`env/environment-ica-new.yml`) | 0.19.1 | newer model families |

Both environments install the same `contextual_drag` package via `pip install -e .[all]` plus the `aiq-magnet` submodule via `pip install -e ./submodules/aiq-magnet`.

## TODO

- **Verify §3 error-conditioning at a larger sample size.** The stock smoke slice (aime24 × 16 problems × n=4) was `FALSIFIED` with only n_kept = 2 surviving the 2F aggregate + verdict filter. Re-run at `n=8` or `max_questions=32` (or with `regime=framing` to bypass the verdict filter) and update the "Last verified" line under §3 once the claim crosses threshold.
- **Verify TED structural drag.** The TED card was skipped in the v0 smoke run. Run `cards/contextual_drag_ted.yaml` end-to-end on 24-game × Qwen3_8B × phase=2f and add a "Last verified" line under "TED structural drag" with the measured `ted_drag` and `n_kept_problems`.
- **Add a `max_tokens` / `context_length` toggle.** Today `max_tokens` in each card is the generation-only budget while `context_length` is the prompt + generation cap on `Qwen3_8B_NoThinking`. Add a card-level switch (e.g. `max_tokens_mode: {rollout_budget | model_max}`) so a card can request "use the model's full advertised context window" (32768 for Qwen3) vs. "cap generation at this rollout token budget" without editing `eval_models_params.json`. When the toggle lands, re-run §2, §3, and TED under the `rollout_budget` setting and refresh the "Last verified" numbers — current results were taken under the model-max-context regime.

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

## Acknowledgements

This deliverable was produced jointly by Princeton-PLI and Kitware as an AIQ-magnet integration for the Contextual Drag evaluation.
