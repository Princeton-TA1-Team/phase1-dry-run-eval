#!/bin/bash
# Root node
python -m cards.nodes.run_drag_experiment \
    --model_config=Qwen3_1.7B_NoThinking \
    --data_path=data/smoke/gpqa/gpqa.ds \
    --init_template_path=prompt_templates/init_response_prompt_templates.json \
    --init_template_key=qa_mc_prompt \
    --twof_template_path=prompt_templates/2f_templates.json \
    --twof_template_key=2f \
    --max_questions=8 \
    --n=8 \
    --max_tokens=2048 \
    --gpu_memory_utilization=0.85 \
    --num_true=0 \
    --num_false=2 \
    --results_fpath=evaluation_runs/3016791f_2026-05-22__23-57-37/kwdagger/drag_experiment/drag_experiment_id_qjboxrbtze2y/results.json 