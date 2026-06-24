# Pre-built §3 (error-conditioning) artefacts — DEFERRED

This directory will eventually contain pre-built artefacts that let
`cards/smoke_runs/Qwen3_8B_NoThinking/error-conditioning-posthoc/aime24.yaml` skip the GPU steps of
its pipeline when the host has no GPU available — specifically:

  * a 16-problem `init_inference/` slice (post-`eval --flatten_dataset`),
  * the `processed_flattened_init_responses.ds` post-postprocess,
  * the `minimal_aggregated_data_T0_F2.ds` post-aggregate,
  * a pre-built `cond_inference/` 2F-evaluated jsonl.

Status: **not built**. Producing these artefacts requires a host with a
GPU large enough to run Qwen3_8B_NoThinking at n=8 on the aime24 slice;
the cards-impl agent that wrote this bundle has no GPU access.

Until they ship, `cards/smoke_runs/Qwen3_8B_NoThinking/error-conditioning-posthoc/aime24.yaml` runs in
its default "inline" mode (everything from clean inference forward is
recomputed on the host's GPU). The card already handles this transparently;
no flag is required.

The cards-impl bot should regenerate this directory after a successful
end-to-end card run on a GPU host by copying:

  ```
  evaluation_runs/error_conditioning/<hash>/direct_inference/evaluated_*.jsonl
  evaluation_runs/error_conditioning/<hash>/direct_inference/evaluated_*_flattened.jsonl
  evaluation_runs/error_conditioning/<hash>/aggregate/minimal_aggregated_data_T0_F2.ds
  evaluation_runs/error_conditioning/<hash>/cond_inference/evaluated_*.jsonl
  ```

into here, plus a `PROVENANCE.md` rolling up source paths + sha256.
