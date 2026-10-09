# compmath

The four competition-mathematics benchmarks of the route-model card, pooled
into one dataset: AIME 2024, AIME 2025, HMMT 2024 and HMMT 2025 (120 problems).

- Built from this repository's own copies: `data/full_data/{aime24,aime25,hmmt24,hmmt25}/*.ds`
  (see their `PROVENANCE.md`).
- Rows are concatenated in that order with no other change, except that `id`
  is namespaced by its source benchmark: `aime24:2`, `hmmt25:13`. Raw ids
  repeat across benchmarks, and the card keys every problem by id.
- This is the id convention of the original draft-composition study, so its
  selected-problem lists can be compared with a card run directly.
