# Pre-built §4 (mitigation) artefacts — DEFERRED

This directory will eventually contain pre-built artefacts that let
`cards/contextual_drag_mitigation.yaml` skip the GPU-bound steps —
specifically:

  * a 16-problem `direct_inference/evaluated_*.jsonl`,
  * the `minimal_aggregated_data_T0_F2.ds` post-aggregate,
  * the `minimal_aggregated_data_T0_F1_flattend_from_F2.ds` (the F1 input
    the mitigation pipeline expects),
  * a pre-built `onef_inference/evaluated_*.jsonl`,
  * the strategy + filter1 stage outputs from a `cm_filter1` run.

Status: **not built**. Producing these artefacts requires a host with a
GPU large enough to run Qwen3_8B_NoThinking at n=8 on the gpqa slice and
to drive the cm_filter1 strategy → filter1 → solve chain; the cards-impl
agent that wrote this bundle has no GPU access.

Until they ship, `cards/contextual_drag_mitigation.yaml` runs in its
default "inline" mode (every step recomputed on the host's GPU). The
card already handles this transparently; no flag is required.

The cards-impl bot should regenerate this directory after a successful
end-to-end card run on a GPU host by copying:

  ```
  evaluation_runs/mitigation/<hash>/direct_inference/evaluated_*.jsonl
  evaluation_runs/mitigation/<hash>/aggregate/minimal_aggregated_data_T0_F2.ds
  evaluation_runs/mitigation/<hash>/aggregate/minimal_aggregated_data_T0_F1_flattend_from_F2.ds
  evaluation_runs/mitigation/<hash>/onef_inference/evaluated_*.jsonl
  evaluation_runs/mitigation/<hash>/mitigation/intermediate/strategy.ds
  evaluation_runs/mitigation/<hash>/mitigation/intermediate/filter1.ds
  evaluation_runs/mitigation/<hash>/mitigation/completions.jsonl
  ```

into here, plus a `PROVENANCE.md` rolling up source paths + sha256.
