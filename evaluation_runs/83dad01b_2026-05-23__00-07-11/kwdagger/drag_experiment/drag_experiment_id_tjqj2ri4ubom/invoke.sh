#!/bin/bash
# Root node
python -m cards.nodes.run_drag_experiment \
    --model_config=Qwen3_8B_NoThinking \
    --data_path=data/smoke/gpqa/gpqa.ds \
    --init_template_path=prompt_templates/init_response_prompt_templates.json \
    --init_template_key=qa_mc_prompt \
    --twof_template_path=prompt_templates/2f_templates.json \
    --twof_template_key=2f \
    --max_questions=8 \
    --n=8 \
    --max_tokens=4096 \
    --gpu_memory_utilization=0.85 \
    --num_true=0 \
    --num_false=2 \
    --results_fpath=evaluation_runs/83dad01b_2026-05-23__00-07-11/kwdagger/drag_experiment/drag_experiment_id_tjqj2ri4ubom/results.json 