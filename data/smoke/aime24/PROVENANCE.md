# `aime24.ds` smoke slice

| Field         | Value                                                                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source        | `/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/neurips_result_aggregation/Contextual-Drag/data/aime24/aime24.ds`                              |
| Source layout | flat `Dataset` (30 rows)                                                                                                                                                    |
| Slice recipe  | `load_from_disk(src).select(range(16))` saved with `save_to_disk`.                                                                                                          |
| Row count     | 16                                                                                                                                                                          |
| Columns       | `problem`, `answer`, `source`, `domain`, `llama8b_solve_rate`, `id`, `label`                                                                                               |
| Synthetic id? | No — `id` is carried from upstream.                                                                                                                                         |
| Canonical artefact | `aime24.ds/data-00000-of-00001.arrow`                                                                                                                                  |
| sha256        | `d2e218a3db84e0711f9bc4eca19b037a3d60d15a7293ce5d17dc255ab387dd82`                                                                                                          |
| Byte size     | 7 376 B                                                                                                                                                                       |
| Transient files dropped | `cache-*.arrow`                                                                                                                                                    |

Used by `cards/contextual_drag_error_conditioning.yaml` (§3) as the
default benchmark. The original `aime24` is the 2024 AIME competition
problem set, repackaged upstream in
`neurips_result_aggregation/Contextual-Drag/data/aime24/`; the slice is
unmodified.
