#!/bin/bash
# Root node
python -m cards.nodes.run_error_conditioning \
    --model_config=Qwen3_1.7B_NoThinking \
    --data_path=data/smoke/aime24/aime24.ds \
    --regime=2f \
    --init_template_path=prompt_templates/init_response_prompt_templates.json \
    --init_template_key=qwen_math_prompt \
    --cond_template_path_1f=prompt_templates/1f_templates.json \
    --cond_template_key_1f=1f \
    --cond_template_path_2f=prompt_templates/2f_templates.json \
    --cond_template_key_2f=2f \
    --max_questions=16 \
    --n=8 \
    --max_tokens=2048 \
    --gpu_memory_utilization=0.85 \
    --results_fpath=evaluation_runs/b83d5f37_2026-05-23__00-00-13/kwdagger/error_conditioning/error_conditioning_id_u94284bl1y0d/results.json 