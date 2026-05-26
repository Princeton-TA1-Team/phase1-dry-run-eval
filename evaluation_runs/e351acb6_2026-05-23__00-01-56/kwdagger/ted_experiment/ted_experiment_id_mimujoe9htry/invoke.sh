#!/bin/bash
# Root node
python -m cards.nodes.run_ted_experiment \
    --model_config=Qwen3_1.7B_NoThinking \
    --data_path=data/smoke/24-game/24-game.ds \
    --init_template_path=prompt_templates/init_response_prompt_templates.json \
    --init_template_key=qwen_math_prompt \
    --twof_template_path=prompt_templates/2f_templates.json \
    --twof_template_key=2f \
    --ted_metric=tree \
    --ted_reduction=min \
    --ted_phase=2f \
    --n_jobs=4 \
    --max_questions=32 \
    --n=8 \
    --max_tokens=2048 \
    --gpu_memory_utilization=0.85 \
    --results_fpath=evaluation_runs/e351acb6_2026-05-23__00-01-56/kwdagger/ted_experiment/ted_experiment_id_mimujoe9htry/results.json 