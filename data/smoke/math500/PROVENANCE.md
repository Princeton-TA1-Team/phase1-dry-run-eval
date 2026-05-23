# `math500.ds` smoke slice

| Field         | Value                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| Source        | `/scratch/gpfs/ARORA/yc6206/multi-llm/Contextual-Drag/data/math500/math500.ds`                       |
| Source layout | `DatasetDict({"test": Dataset(500 rows)})`                                                            |
| Slice recipe  | `load_from_disk(src)["test"].select(range(4))` → flat `Dataset` saved with `save_to_disk`.            |
| Row count     | 4                                                                                                     |
| Columns       | `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`                                      |
| Synthetic id? | No — `unique_id` carried from upstream; downstream pipelines that require `id` expect the inference step to derive one from `unique_id` (the smoke card's `inference run` does this implicitly via PR-2's id-derivation logic). |
| Canonical artefact | `math500.ds/data-00000-of-00001.arrow`                                                           |
| sha256        | `3ac9286a94f22e7283e2966fdfe245f222b9c3feff1c21f3c4dd4154128b105c`                                    |
| Byte size     | 4 120 B                                                                                               |
| Transient files dropped | `cache-*.arrow` (HF leaves them after `.select`; we delete to keep the bundle deterministic). |

This slice exists so that `cards/contextual_drag_smoke.yaml` can run end-
to-end on a fresh checkout without depending on a per-user dataset path.
The original `math500.ds` is a public benchmark (Hendrycks et al., 2021,
"MATH"); the upstream copy on `/scratch/gpfs/ARORA/yc6206/` is unmodified
relative to that release.
