# Source ladders

The five already-characterized models of the route-model card. The card
measures one new target model (GPT-OSS-20B) and takes these as its sources:
they fix the shared exponent gamma, and the target never enters their fit.

| File | Model | Eligible problems |
|---|---|---|
| `GPT_OSS_120B.json` | openai/gpt-oss-120b | 30 |
| `Gemma4_31B.json` | google/gemma-4-31B-it | 20 |
| `Gemma4_E4B.json` | google/gemma-4-E4B-it | 22 |
| `Qwen3_32B.json` | Qwen/Qwen3-32B | 22 |
| `Qwen3_8B.json` | Qwen/Qwen3-8B | 18 |

**Protocol.** These are the same measurements the card makes, collected with
the same protocol before submission (contextual-drag study, draft-composition
arm D, 2026-09-16/17). Problems from AIME/HMMT 2024-25 were kept when at least 4
of 16 clean answers were right and at least 4 wrong. For T = 0..4, one draft
set of T correct and 4 - T wrong self-written drafts was shown in 8 distinct
orders, with the `dc_same_math` prompt, thinking on, a budget of up to 32768
tokens, sampling seed 42 and composition seed 20260916. The baseline q is a
fresh 16-sample resample under seed 1234, with an ungradable answer counted
wrong. Wrong-answer classes use `norm_answer`.

**Format.** `contextual_drag.analysis.route_model.io`, schema
`contextual_drag.route_model.ladder/1`.

**Built by** `analysis/route_model_retro/export_ladders.py` in the study
repository, from `data/results/dc_D_T{T}F{4-T}/compmath_<model>`,
`data/results/direct_fresh/<task>_<model>`, and
`data/draft_composition/MANIFEST_armD_compmath.json`.
