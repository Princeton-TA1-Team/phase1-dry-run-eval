# `24-game.ds` smoke slice

| Field         | Value                                                                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source        | `/scratch/gpfs/ARORA/yc6206/metacognition_eval/data/24-game/24-game.ds`                                                                                                     |
| Source layout | flat `Dataset` (1362 rows)                                                                                                                                                  |
| Slice recipe  | `load_from_disk(src).select(range(32))` then `map(lambda ex, idx: {**ex, "id": f"24-game_{idx:04d}"}, with_indices=True)` to add a synthetic id; saved with `save_to_disk`. |
| Row count     | 32                                                                                                                                                                          |
| Columns       | `problem`, `answer`, `solutions`, `compute_graph`, `numbers`, `source`, `domain`, `llama8b_solve_rate`, `id` (synthetic), `label`                                          |
| Synthetic id? | **Yes** — the upstream `24-game.ds` lacks an `id` column; downstream pipelines (`data initial-sampling-postprocess`, `data aggregate`, mitigation `pipeline.aggregate`) all key on `id`, so we mint one from the row index at slice time: `id = f"24-game_{idx:04d}"`. |
| Canonical artefact | `24-game.ds/data-00000-of-00001.arrow`                                                                                                                                |
| sha256        | `a933eec02c8e9fbccd786030c88dafdc56cbd00b73a5932d934a15cd3eb26714`                                                                                                          |
| Byte size     | 33 784 B                                                                                                                                                                      |
| Transient files dropped | `cache-*.arrow`                                                                                                                                                    |

Used by `cards/contextual_drag_ted.yaml` (TED) as the default benchmark.
The original `24-game` is the OpenAI "24-game" arithmetic-puzzle dataset,
repackaged upstream in `yc6206/metacognition_eval`. We slice the first 32
rows and add a synthetic `id`; problem content is unmodified.
