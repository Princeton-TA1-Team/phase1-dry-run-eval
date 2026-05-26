# `gpqa.ds` smoke slice

| Field         | Value                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| Source        | `/scratch/gpfs/ARORA/yc6206/metacognition_eval/data/gpqa/gpqa.ds`                                      |
| Source layout | flat `Dataset` (198 rows)                                                                              |
| Slice recipe  | `load_from_disk(src).select(range(16))` saved with `save_to_disk`.                                     |
| Row count     | 16                                                                                                     |
| Columns       | `problem`, `answer`, `source`, `domain`, `llama8b_solve_rate`, `id`, `label`                          |
| Synthetic id? | No — `id` is carried from upstream.                                                                    |
| Canonical artefact | `gpqa.ds/data-00000-of-00001.arrow`                                                              |
| sha256        | `d61acea293e1f1d97569eb62539c483e4d6d3914859d2172fc418e65a279a8e3`                                     |
| Byte size     | 12 888 B                                                                                                |
| Transient files dropped | `cache-*.arrow`                                                                              |

Used by `cards/contextual_drag.yaml` (baseline drag claim) and
`cards/contextual_drag_mitigation.yaml` (§4) as the default benchmark.
The original `gpqa` (Rein et al., 2023, "GPQA: A Graduate-Level
Google-Proof Q&A Benchmark") was repackaged with multiple-choice labels
upstream in `yc6206/metacognition_eval`; the slice is unmodified.
