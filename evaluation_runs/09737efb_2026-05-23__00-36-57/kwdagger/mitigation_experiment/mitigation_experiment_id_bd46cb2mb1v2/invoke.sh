#!/bin/bash
# Root node
python -m cards.nodes.run_mitigation_experiment \
    --model_config=Qwen3_8B_NoThinking \
    --data_path=data/smoke/gpqa/gpqa.ds \
    --variant=cm_filter1 \
    --init_template_path=prompt_templates/init_response_prompt_templates.json \
    --init_template_key=qa_mc_prompt \
    --onef_template_path=prompt_templates/1f_templates.json \
    --onef_template_key=1f \
    --mit_template_path=prompt_templates/context_manipulation_templates.json \
    --max_questions=16 \
    --n=8 \
    --n_samples_solve=8 \
    --max_tokens=8192 \
    --gpu_memory_utilization=0.85 \
    --results_fpath=evaluation_runs/09737efb_2026-05-23__00-36-57/kwdagger/mitigation_experiment/mitigation_experiment_id_bd46cb2mb1v2/results.json 