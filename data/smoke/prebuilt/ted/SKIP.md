# Pre-built TED artefacts — DEFERRED

This directory will eventually contain pre-built artefacts that let
`cards/contextual_drag_ted.yaml` skip the GPU-bound steps — specifically:

  * a 32-problem `direct_inference/evaluated_*_flattened.jsonl`,
  * the `minimal_aggregated_data_T0_F2.ds` post-aggregate,
  * a pre-built `twof_inference/evaluated_*_flattened.jsonl`,
  * a pre-computed TED `cache.json` for `(phase=2f, metric=tree)` keyed
    by `Qwen3_8B_NoThinking`.

Status: **not built**. Producing these artefacts requires a host with a
GPU large enough to run Qwen3_8B_NoThinking at n=8 on the 24-game slice;
the cards-impl agent that wrote this bundle has no GPU access.

Until they ship, `cards/contextual_drag_ted.yaml` runs in its default
"inline" mode (everything from clean inference forward is recomputed on
the host's GPU; the TED build-cache step itself is CPU-bound but cheap
for a 32-problem slice — single-digit minutes). The card already handles
this transparently; no flag is required.

The cards-impl bot should regenerate this directory after a successful
end-to-end card run on a GPU host by copying:

  ```
  evaluation_runs/ted/<hash>/direct_inference/evaluated_*_flattened.jsonl
  evaluation_runs/ted/<hash>/aggregate/minimal_aggregated_data_T0_F2.ds
  evaluation_runs/ted/<hash>/twof_inference/evaluated_*_flattened.jsonl
  evaluation_runs/ted/<hash>/ted_cache.json   ->   cache/2f/Qwen3_8B_NoThinking.json
  ```

into here, plus a `PROVENANCE.md` rolling up source paths + sha256.
