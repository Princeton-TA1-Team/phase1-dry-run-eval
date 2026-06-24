# `data/smoke/` — smoke-data bundle

Workspace-relative datasets that bootstrap the five magnet cards in
`cards/` for a fresh checkout. Total payload ≈ 60 KB of arrow data
(actual; the headline "27 MB total" in the plan was an overestimate
because we slice tightly and the pre-built artefacts are deferred).

Every card defaults to a `data/<task>/<task>.ds` path so the cards
run end-to-end on a single 24 GB GPU without needing the user to set
`AIQ_CDRAG_DATA_ROOT` or pass an explicit `--data_path`.

## Bundled raw slices

| Slice                 | Source                                                                                                                                                                       | Split    | Rows | Arrow shard                                | sha256          | Bytes  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---: | ------------------------------------------ | --------------- | ------ |
| `math500/math500.ds`  | `/scratch/gpfs/ARORA/yc6206/multi-llm/Contextual-Drag/data/math500/math500.ds`                                                                                              | `test`   | 4    | `math500.ds/data-00000-of-00001.arrow`     | `3ac9286a94f2…` | 4 120  |
| `gpqa/gpqa.ds`        | `/scratch/gpfs/ARORA/yc6206/metacognition_eval/data/gpqa/gpqa.ds`                                                                                                            | (flat)   | 16   | `gpqa.ds/data-00000-of-00001.arrow`        | `d61acea293e1…` | 12 888 |
| `aime24/aime24.ds`    | `/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/neurips_result_aggregation/Contextual-Drag/data/aime24/aime24.ds`                               | (flat)   | 16   | `aime24.ds/data-00000-of-00001.arrow`      | `d2e218a3db84…` | 7 376  |
| `24-game/24-game.ds`  | `/scratch/gpfs/ARORA/yc6206/metacognition_eval/data/24-game/24-game.ds`                                                                                                      | (flat)   | 32   | `24-game.ds/data-00000-of-00001.arrow`     | `a933eec02c8e…` | 33 784 |

Full sha256 + slice recipe in each `<task>/PROVENANCE.md`.

## Slice recipe (uniform)

```python
from datasets import load_from_disk
ds = load_from_disk(SOURCE)
if SPLIT:                                 # math500 only
    ds = ds[SPLIT]
sliced = ds.select(range(N_ROWS))
if name == "24-game":                     # 24-game.ds lacks `id` upstream
    sliced = sliced.map(
        lambda ex, idx: {**ex, "id": f"24-game_{idx:04d}"},
        with_indices=True,
    )
sliced.save_to_disk(DEST)                 # drops cache-*.arrow afterwards
```

## Card → slice → claim mapping

| Card                                             | Slice                          | Claim symbol           | Threshold |
| ------------------------------------------------ | ------------------------------ | ---------------------- | --------- |
| `smoke_runs/Qwen3_8B_NoThinking/wiring/math500.yaml`                     | `math500/math500.ds` (4 rows)  | `accuracy`             | ≥ 0.25    |
| `smoke_runs/Qwen3_8B_NoThinking/drag/gpqa.yaml`                           | `gpqa/gpqa.ds` (16 rows)       | `drag`                 | ≥ 0.05    |
| `smoke_runs/Qwen3_8B_NoThinking/error-conditioning-posthoc/aime24.yaml`        | `aime24/aime24.ds` (16 rows)   | `delta_acc`            | ≥ 0.05    |
| `smoke_runs/Qwen3_8B_NoThinking/mitigation/gpqa.yaml`                | `gpqa/gpqa.ds` (16 rows)       | `recovery_rate`        | ≥ 0.20    |
| `smoke_runs/Qwen3_8B_NoThinking/ted/24-game.yaml`                       | `24-game/24-game.ds` (32 rows) | `ted_drag`             | ≥ 1.0     |

`smoke_runs/Qwen3_8B_NoThinking/drag/gpqa.yaml` and `smoke_runs/Qwen3_8B_NoThinking/mitigation/gpqa.yaml` both default
to gpqa — that's deliberate (same model × benchmark cell, different
claim). The smoke card uses math500 because it's in the model's "easy"
zone so wiring failures stand out, while the real drag claim needs the
"ambiguous" zone (gpqa for Qwen3_8B) to land a measurable Δ.

## Deferred

Pre-built artefacts under `prebuilt/{error_conditioning,mitigation,ted}/`
are **not shipped** in this bundle. Each subdir contains a `SKIP.md`
explaining why and what to regenerate (the cards-impl agent that built
this bundle has no GPU access; producing the pre-built artefacts requires
running the corresponding card's GPU pipeline once on a real host).

The cards already work in "inline" mode (every step recomputed on the
host's GPU at card-run time) without these pre-built artefacts; they
exist purely as a performance optimisation for CI runs on GPU-equipped
runners. Until they ship:

```
data/smoke/prebuilt/error_conditioning/SKIP.md   # placeholder
data/smoke/prebuilt/mitigation/SKIP.md           # placeholder
data/smoke/prebuilt/ted/SKIP.md                  # placeholder
```

No `MISSING.md` files were needed — every requested raw slice was
locatable under `/scratch/gpfs/ARORA/`.
