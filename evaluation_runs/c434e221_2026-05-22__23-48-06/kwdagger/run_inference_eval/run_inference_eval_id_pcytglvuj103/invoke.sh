#!/bin/bash
# Root node
python -m cards.nodes.run_inference_eval \
    --model_config=Qwen3_1.7B_NoThinking \
    --data_path=data/math500/math500.ds \
    --prompt_template_path=prompt_templates/init_response_prompt_templates.json \
    --prompt_template_key=qwen_math_prompt \
    --max_questions=4 \
    --n=1 \
    --max_tokens=2048 \
    --gpu_memory_utilization=0.85 \
    --results_fpath=evaluation_runs/c434e221_2026-05-22__23-48-06/kwdagger/run_inference_eval/run_inference_eval_id_pcytglvuj103/results.json 