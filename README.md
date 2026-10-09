# Contextual Drag in GPT-OSS-20B: Route-Model Prediction

**Princeton TA1 (Princeton-2) — Phase I Final evaluation card for DARPA AIQ**

This repository contains the evaluation card `cards/contextual_drag_ccg_transfer.yaml`, which implements *Phase I Evaluation Plan, Team Princeton-2* for **GPT-OSS-20B**. The card measures how GPT-OSS-20B's accuracy changes when its own earlier drafts are in its context. It then tests whether a route model of contextual drag predicts that accuracy from a partial measurement. The card runs on [AIQ-MAGNET](https://github.com/AIQ-Kitware/aiq-magnet) 0.1.0 and is started with `bash scripts/run_ccg_transfer.sh`.

The earlier Phase I dry-run cards and the setup they ran on are documented in [README_LEGACY.md](README_LEGACY.md).

## Contents

1. [Motivation](#1-motivation)
2. [Route Model of Contextual Drag](#2-route-model-of-contextual-drag)
3. [Evaluation Protocol](#3-evaluation-protocol)
4. [Metrics and Pass Criterion](#4-metrics-and-pass-criterion)
5. [Pre-Submission Result](#5-pre-submission-result)
6. [Running the Evaluation](#6-running-the-evaluation)
7. [Repository Structure](#7-repository-structure)
8. [Testing](#8-testing)

## 1. Motivation

Asking a language model to improve an earlier attempt in its context is a basic building block of recursive self-improvement. Iterative agentic workflows assume that a model can use previous attempts to produce a better answer, guided by feedback from itself, tools or other agents. An earlier attempt in context does more than add information, however. Correct attempts guide the model toward the right solution, and incorrect attempts bias it toward similar errors. This latter effect is *contextual drag* [1].

The evaluation uses a simple theory that predicts a model's accuracy from the composition of the drafts in its context. It asks whether GPT-OSS-20B, measured only partially, behaves as the theory predicts once its parameters are set from those partial measurements.

## 2. Route Model of Contextual Drag

The final answer is treated as a choice among routes. The model either adopts one of the answers in its context or sets the drafts aside and solves the problem independently, and each route is taken with probability proportional to its weight.

Consider $`K \ge 1`$ drafts of which $`T`$ are correct. Let $`q_{i}`$, the *baseline accuracy*, be the model's probability of solving problem $`i`$ without drafts. Correct drafts share one normalized answer. Incorrect drafts fall into answer classes with multiplicities $`\lambda = (n_1, \dots, n_r)`$, where $`\sum_j n_j = K - T`$. The route model gives the probability of a correct final answer as

```math
F_K(T, \lambda, q_{i}; \theta) = \frac{a\, T^{\gamma} + q_{i}}{a\, T^{\gamma} + b \sum_j n_j^{\gamma} + 1}, \qquad \theta = (a, b, \gamma). \qquad\qquad (1)
```

The parameters are:
- **Draft weights.** The positive weights $`a`$ and $`b`$ weigh a single correct and a single incorrect draft against the independent solution, which has unit weight and is correct with probability $`q_{i}`$.
- **Agreement exponent.** $`\gamma \ge 0`$ describes how agreement compounds. At $`\gamma = 1`$ a repeated answer counts in proportion to its repetitions, above 1 agreement is amplified, and at 0 only distinct answers matter.
- **Absent routes** have zero weight.

The model assumes that parameters are common across problems, and that the independent-solution route retains accuracy $`q_{i}`$ in context.

**Proposition 1 (Transfer sensitivity).** Fix an integer $`K \ge 1`$ and a context $`(T, \lambda)`$ with $`T \in \{0, \dots, K\}`$. For $`q, \hat q \in [0, 1]`$, positive $`a, b, \hat a, \hat b`$ and nonnegative $`\gamma, \hat\gamma`$, let $`F_K`$ and $`\hat F_K`$ denote (1) at the two parameter sets. Then

```math
|\hat F_K - F_K| \le |\hat q - q| + \tfrac{1}{4}\left( \left|\log \tfrac{\hat a}{a}\right| + \left|\log \tfrac{\hat b}{b}\right| + \log(K)\, |\hat\gamma - \gamma| \right).
```

Averaging over problems gives the same bound with the mean absolute error in $`q_i`$. The bound omits error in the route model itself and sampling error, so the evaluation measures prediction error on held-out outcomes.

The route model, Proposition 1 and the claim the card tests are recorded with their premises in `theory/indexes/route_model.yaml`. The card links to them through MAGNET's theory annotations.

## 3. Evaluation Protocol

### 3.1 Setting

The evaluation uses four competition mathematics benchmarks, AIME 2024, AIME 2025, HMMT 2024 and HMMT 2025: 120 problems in total (`data/full_data/compmath`). Every context holds $`K = 4`$ drafts written by GPT-OSS-20B itself, and $`T`$ ranges from 0 to 4.

**Fixed before submission.** Model version, prompts, inference settings, evaluation code, fitting settings, random seeds and the pass criterion:

| Setting | Value |
|---|---|
| Model | `openai/gpt-oss-20b` (`GPT_OSS_20B` in `src/contextual_drag/resources/inference/eval_models_params.json`) |
| Sampling | Temperature 1.0, top-p 1.0, top-k 40; reasoning on; up to 32,768 tokens per response |
| Prompts | `qwen_math_prompt` for answers without drafts; `dc_same_math` (`prompt_templates/draft_composition_templates.json`) for answers with drafts |
| Seeds | Draft pool 42; contexts 42; baseline 1234; draft composition 20260916; bootstrap 20260904 |
| Pass criterion | Errors at $`T = 2`$ and $`T = 3`$ both at most 5.0 percentage points |

All of these are set in the card's `kwdagger.matrix`.

### 3.2 Measurement under Contextual Drag

GPT-OSS-20B first solves each problem 16 times with nothing else in its context. These 16 answers become its drafts.
- **Problem selection.** A problem is kept only if at least four of the 16 answers are correct and at least four are wrong. Every mix of four drafts, from all wrong to all correct, can then be built on the same problems. These are also the problems that matter for iterative self-improvement: the model often fails on its first try but can still reach the correct answer with more attempts.
- **Contexts.** For each kept problem and each value of $`T`$, the evaluation picks $`T`$ correct and $`4 - T`$ wrong drafts. It shows them in eight distinct random orders. The model is not told which drafts are correct and is not asked to judge them; it is asked only to solve the problem.
- **Measured accuracy.** The final answer of every draft and the number of drafts sharing each wrong answer are recorded. The model's new answers are graded and averaged, first over the eight orders and then over problems, giving the measured accuracy $`p^{\mathrm{obs}}_{T}`$.
- **Baseline accuracy.** $`q_{i}`$ is estimated from 16 new answers per problem, drawn under a different seed. The first 16 answers are not reused, because they chose the problems, and reusing them would make the baseline look too high.

**Grading and draft text.** An answer is the last `\boxed{}` expression in the response, compared with the reference by `math_verify`. A response without one counts as incorrect. Drafts are each answer's final section with the reasoning trace removed. Wrong-answer classes group drafts by their normalized answer; an unparsable wrong draft forms a class of its own.

### 3.3 Prediction for GPT-OSS-20B

**Shared exponent.** The exponent $`\gamma`$ is shared across models. It is estimated by maximum likelihood from reference measurements of five other models, collected under this same protocol before submission. Those measurements are bundled in `data/route_model/source_ladders` with their provenance; the card reads them and does not regenerate them.

**Calibration.** For GPT-OSS-20B, only three quantities are used: its baseline accuracy, and its accuracy with zero and with one correct draft. From these, its weights $`(a, b)`$ are fitted by maximum likelihood with $`\gamma`$ fixed. Each fit uses every problem's baseline accuracy and the actual pattern of wrong answers in each context.

**Prediction.** Equation (1) then predicts GPT-OSS-20B's accuracy with two, three and four correct drafts. If the calibration data imply a negative weight, the prediction is reported as failed.

**Why not calibrate at $`T = 4`$.** The accuracy with four correct drafts is not used to set the weights. GPT-OSS-20B scores above 98% in that case, so the accuracy barely depends on $`a`$: very different values of $`a`$ give almost the same accuracy. Working backward from it would turn a one-point measurement error into an error that can double or halve $`a`$.

## 4. Metrics and Pass Criterion

For each predicted value of $`T`$ the card reports the error

```math
\texttt{ccg\_abs\_error\_pp} = 100\, |\hat p_{T} - p^{\mathrm{obs}}_{T}|,
```

together with the signed error, the measured and predicted accuracies, and the fitted parameters.

Following the BAA criterion of at most 5% error, the card passes if the $`T = 2`$ and $`T = 3`$ errors are both at most 5.0 points. The $`T = 4`$ prediction is reported separately and does not enter the criterion.

MAGNET reports the outcome of the claim as follows:

| MAGNET result | Condition |
|---|---|
| VERIFIED | Both deciding errors are at most 5.0 points. |
| FALSIFIED | A deciding error exceeds 5.0 points, or the calibration data imply a negative weight. |
| INCONCLUSIVE | GPT-OSS-20B could not be measured: no problem met the selection rule, more than 5% of the kept problems lack a full set of contexts or a baseline, or a stage of the pipeline failed. |

### 4.1 Uncertainty and Sample Size

Simultaneous 95% intervals for the signed errors come from a bootstrap over problems with 2,000 replicates. Each replicate refits every stage, including the shared exponent. The intervals are Bonferroni-adjusted: the $`T = 2`$ and $`T = 3`$ errors form one family and the $`T = 4`$ error another. The result has statistical support if both deciding intervals lie inside $`[-5, 5]`$; otherwise the intervals are reported as unresolved.

GPT-OSS-20B's selection rule kept 22 problems in pre-submission data. At that size the intervals are wider than five points, so the pass criterion rests on point estimates. Five-point half-widths would need roughly 50 to 110 eligible problems.

## 5. Pre-Submission Result

The same procedure on GPT-OSS-20B measurements collected before submission (22 eligible problems, baseline accuracy 56.8%):

| $`T`$ | Measured | Predicted | Signed error (pp) | 95% interval |
|---|---|---|---|---|
| 2 | 90.3% | 87.0% | −3.31 | −17.2 to +6.9 |
| 3 | 96.6% | 94.0% | −2.60 | −13.3 to +4.4 |
| 4 | 98.3% | 96.8% | −1.53 | −7.2 to +2.4 |

Both deciding errors are within 5.0 points, so MAGNET reports VERIFIED. A fresh run of the card draws new answers. Given these interval widths, its errors may fall on either side of the tolerance.

## 6. Running the Evaluation

### 6.1 Installation

```bash
git clone https://github.com/Princeton-TA1-Team/phase1-dry-run-eval.git
cd phase1-dry-run-eval
git checkout phase1-final/ccg-transfer
bash scripts/install.sh
conda activate phase1-dry-run-eval
```

`scripts/install.sh` creates the conda environment `phase1-dry-run-eval` with Python 3.11, vLLM 0.10.2, MAGNET 0.1.0 and this package. In an existing Python 3.11 environment, the equivalent is:

```bash
pip install -e ".[inference,magnet,analysis,eval]"
```

All data, prompts and reference measurements are in the repository. Only the model weights and tokenizer are downloaded at run time.

### 6.2 Running the Card

```bash
bash scripts/run_ccg_transfer.sh --dry_run True    # compile the ten stages and check the setup
bash scripts/run_ccg_transfer.sh                   # full evaluation
```

The full evaluation generates about 3,000 to 3,600 responses, each up to 32,768 tokens with reasoning on, which takes hours of GPU time. The pre-submission data was generated on 80 GB GPUs. Results are written to `evaluation_runs/` ([Section 6.4](#64-output-files)).

| Variable | Default | Purpose |
|---|---|---|
| `OUTPUT_PATH` | `evaluation_runs` | Result directory |
| `BACKEND` | `serial` | `serial` runs the stages in the foreground; `tmux` runs them in tmux sessions |
| `CONTEXTUAL_DRAG_ENDPOINT` | unset | OpenAI-compatible endpoint; unset means the local GPU ([Section 6.3](#63-model-serving)) |

Further arguments go to `magnet evaluate_new`. The script runs the equivalent of:

```bash
PYTHONPATH=$PWD:$PWD/src magnet evaluate_new cards/contextual_drag_ccg_transfer.yaml \
    --output_path evaluation_runs --backend serial
```

**Containerized execution.** The `Dockerfile` builds an image with the same dependencies. MAGNET can run each stage in that image:

```bash
docker build -t contextual-drag-gpu .
PYTHONPATH=$PWD:$PWD/src magnet evaluate_new cards/contextual_drag_ccg_transfer.yaml \
    --output_path evaluation_runs --backend tmux \
    --container_image contextual-drag-gpu
```

### 6.3 Model Serving

Each generation stage either builds an in-process vLLM engine on one GPU, or sends requests to an OpenAI-compatible endpoint.

**Endpoint configuration.**
- **Address.** Set `CONTEXTUAL_DRAG_ENDPOINT` to the endpoint's base URL ending in `/v1`, and `CONTEXTUAL_DRAG_ENDPOINT_API_KEY` if one is required. Under infer-stack leasing, `OPENAI_BASE_URL` is used instead.
- **Model.** The endpoint must serve the model as `openai/gpt-oss-20b`.
- **Server.** It should be vLLM-compatible, because requests carry `top_k`, a per-request `seed` and `skip_special_tokens: false`.

**Context window.** The served context window (`max_model_len`) must be at least 32,768 tokens, matching the conditions of the reference measurements. Each generation stage checks this through `GET /v1/models` and stops if the window is smaller, giving an INCONCLUSIVE result. When a long prompt leaves less room than 32,768 tokens, the request's budget is capped at the remaining window, which is where an in-process engine stops.

**Reproducibility notes.**
- **Checkpoint.** The pre-submission data used `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`. The package does not pin the revision, so this revision should be served or cached.
- **Prompt date.** The GPT-OSS chat template writes the current date into every prompt.

**Interruption.** Each stage records its outputs row by row and resumes after an interruption.

### 6.4 Output Files

Paths are relative to `--output_path`:

| Path | Contents |
|---|---|
| `<hash>_<time>/verdict.json` | Overall result |
| `<hash>_<time>/results/<id>/verdict.json` | Claim output, including the two deciding errors |
| `<hash>_<time>/theory.json` | Statements the card tests and the premises it assumes or satisfies |
| `_kwdagger/route_loo/<id>/results.json` | Errors, intervals and fitted parameters |
| `_kwdagger/route_loo/<id>/route_model_loo.json` | Complete analysis, including every bootstrap interval |
| `_kwdagger/ladder/<id>/ladder.json` | GPT-OSS-20B's measurements under contextual drag |

The run log prints the deciding errors on a line beginning `[ccg_transfer]`.

## 7. Repository Structure

Components used by the card:

```
cards/
├── contextual_drag_ccg_transfer.yaml   evaluation card: claim, settings, theory links
├── pipelines.py                        route_model_pipeline(): ten-stage kwdagger pipeline
└── nodes/
    ├── cd_seeded_inference.py          generation round with declared seed and endpoint check
    ├── cd_eval.py, cd_postprocess.py   grading; removal of reasoning traces from drafts
    ├── cd_compose.py                   problem selection and draft contexts
    ├── cd_ladder.py                    assembly of GPT-OSS-20B's measurements
    └── cd_route_loo.py                 estimation, prediction and bootstrap
src/contextual_drag/
├── analysis/route_model/               route model, maximum-likelihood fits, prediction, bootstrap
├── data/draft_composition.py           draft-context construction
├── inference/                          vLLM and endpoint generation with per-row resumption
└── evaluation/math/                    answer extraction and equivalence
data/
├── full_data/compmath/                 AIME and HMMT 2024–25, 120 problems
└── route_model/source_ladders/         reference measurements that fix the shared exponent
prompt_templates/draft_composition_templates.json
theory/indexes/route_model.yaml
scripts/install.sh, scripts/run_ccg_transfer.sh
Dockerfile
```

All other components belong to the earlier dry-run cards.

## 8. Testing

```bash
pip install -e ".[magnet,analysis,eval,dev]"     # vLLM is stubbed in the tests
PYTHONPATH=$PWD:$PWD/src pytest tests/ -q
```

`tests/test_route_model.py` covers the route model and its estimation:
- **Fits.** Maximum-likelihood fits recover known parameters.
- **Prediction.** Predictions are exact when the route model holds.
- **Failure handling.** An implied negative weight fails the prediction.
- **Bootstrap.** Resampling is seeded.
- **Measurement files.** The file format round-trips.

`tests/test_cards.py` checks the schema and claim of every card.

## References

[1] Yun Cheng, Xingyu Zhu, Haoyu Zhao and Sanjeev Arora. *Contextual Drag: How Errors in the Context Affect LLM Reasoning.* arXiv:2602.04288, 2026.

```bibtex
@article{cheng2026contextual,
  title         = {Contextual Drag: How Errors in the Context Affect {LLM} Reasoning},
  author        = {Cheng, Yun and Zhu, Xingyu and Zhao, Haoyu and Arora, Sanjeev},
  journal       = {arXiv preprint arXiv:2602.04288},
  year          = {2026},
  eprint        = {2602.04288},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.04288}
}
```

## Contact

Yun Cheng (yc6206@princeton.edu)

## Acknowledgements

Developed by the Princeton TA1 team. The card builds on Kitware's MAGNET 0.1.0 migration of this repository.
