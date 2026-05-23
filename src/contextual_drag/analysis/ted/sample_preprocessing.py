"""Build per-problem (anchored, init) edit-distance evaluation entries."""

from __future__ import annotations

import ast
import json
import re

from joblib import Parallel, delayed
from tqdm import tqdm

from contextual_drag.evaluation.math.utils.math_utils import extract_boxed_answer

from contextual_drag.analysis.ted.edit_distances import _parse_expr
from contextual_drag.analysis.ted.utils import (
    contains_textual_latex,
    normalize_latex_to_infix,
)


def parse_verification_output(output):
    if output is None or len(output.strip()) == 0:
        return None
    output = output.lower().replace('*', '').replace('\n', '')

    incorrect_keywords = [
        "draft solution is incorrect", "draft is incorrect", "draft incorrect",
        "\\boxed{\\text{incorrect}}",
    ]
    correct_keywords = [
        "draft solution is correct", "draft is correct", "draft correct",
        "\\boxed{\\text{correct}}",
    ]

    for kw in correct_keywords:
        if kw in output:
            return True
    for kw in incorrect_keywords:
        if kw in output:
            return False

    pattern = r"<overall_verdict>\s*([^\n]*)</overall_verdict>"
    verdict_results = re.findall(pattern, output, re.DOTALL)
    verdict_result = verdict_results[-1].strip() if verdict_results else None
    if verdict_result is None:
        return None
    return verdict_result.lower() == "correct"


def parse_extracted_answer(extracted_answer):
    if extracted_answer is None or len(extracted_answer.strip()) == 0:
        return None
    extracted_answer = extracted_answer.replace('\\displaystyle', '')
    if contains_textual_latex(extracted_answer):
        return None
    if len(extracted_answer) > 50:
        return None
    if extracted_answer[0] == '(' and extracted_answer[-1] == ')':
        extracted_answer = extracted_answer[1:-1]
    try:
        extracted_formula = normalize_latex_to_infix(
            extracted_answer.replace('×', '*').replace('÷', '/').split('=')[0].strip()
        )
        extracted_formula = re.sub(r'[a-zA-Z]', '', extracted_formula)
        _parse_expr(extracted_formula)
        return extracted_formula
    except Exception:
        return None


def _extract_from_text_fallback(text: str) -> str | None:
    """Last-resort traj answer extraction for models that don't use \\boxed{}."""
    for line in reversed(text.split('\n')):
        line = line.strip().rstrip('`').rstrip('.')
        if not line or len(line) > 80:
            continue
        ans = parse_extracted_answer(line)
        if ans is not None:
            return ans
    return None


def preprocess_anchored_entry(entry, num_trajs=1, iteration=None):
    problem_id = entry['id']
    if not iteration:
        traj_generated_responses = []
        for i in range(1, num_trajs + 1):
            meta_ans = entry[f'traj{i}_metadata']['init_response_generations_extracted_answer']
            if meta_ans is not None:
                traj_generated_responses.append(meta_ans)
            else:
                traj_text = entry.get(f'traj{i}', '') or ''
                traj_generated_responses.append(_extract_from_text_fallback(traj_text))
        traj_extracted_answers = str([
            parse_extracted_answer(t) for t in traj_generated_responses
        ])
        generation_column = 'init_response_generations'
    else:
        traj_generated_responses = [
            extract_boxed_answer(entry[f'traj{i}']) for i in range(1, num_trajs + 1)
        ]
        traj_extracted_answers = str(traj_generated_responses)
        generation_column = f'round{iteration}_response_generations'

    if all(t is None for t in ast.literal_eval(traj_extracted_answers)):
        return None

    anchored_generation = entry[f'{generation_column}_extracted_answer']
    finish_reason = entry[f'{generation_column}_finish_reason']
    if finish_reason != 'stop':
        return None
    anchored_raw_response = entry[f'{generation_column}_generated_response']
    anchored_extracted_answer = parse_extracted_answer(anchored_generation)
    if anchored_extracted_answer is None:
        return None

    if '<think>' in anchored_raw_response:
        anchored_nonthinking_output = anchored_raw_response.split('<think>')[1].split('</think>')[0].strip()
    else:
        anchored_nonthinking_output = anchored_raw_response.split('assistantfinal')[-1].strip()
    anchored_generation_verdict = parse_verification_output(anchored_nonthinking_output)
    anchored_generation_correctness = entry[f'{generation_column}_correctness']
    metadata = [
        anchored_extracted_answer,
        {
            "finish_reason": entry[f'{generation_column}_finish_reason'],
            "correctness": anchored_generation_correctness,
            "verdict": anchored_generation_verdict,
        },
    ]
    return (problem_id, traj_extracted_answers, metadata)


def _expand_list_format(row: dict) -> list[dict]:
    """Expand a list-format row into one flat dict per generation."""
    gens = row.get("init_response_generations", [])
    if not gens:
        return []
    base = {k: v for k, v in row.items() if k != "init_response_generations"}
    result = []
    for gen in gens:
        flat = dict(base)
        flat["init_response_generations_extracted_answer"] = gen.get("extracted_answer")
        flat["init_response_generations_finish_reason"]    = gen.get("finish_reason")
        flat["init_response_generations_correctness"]      = gen.get("correctness")
        flat["init_response_generations_generated_response"] = gen.get("generated_response")
        flat["init_response_generations_response_id"]      = gen.get("response_id")
        result.append(flat)
    return result


def build_processed_anchored_data(path, n_jobs=8, num_trajs=1, iteration=None, fmt="flat"):
    with open(path, 'r') as f:
        raw = [json.loads(line) for line in f]

    if fmt == "list":
        data = [flat for row in raw for flat in _expand_list_format(row)]
    else:
        data = raw

    # Threading backend: SymPy interning on pathological inputs grows large
    # heaps and SIGKILLs loky workers. Threading shares one heap so memory
    # blowup stays bounded.
    preprocessed = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(preprocess_anchored_entry)(entry, num_trajs=num_trajs, iteration=iteration)
        for entry in tqdm(data)
    )

    processed_data = {}
    for item in preprocessed:
        if item is None:
            continue
        problem_id, traj_extracted_answers, anchored_metadata = item
        if problem_id not in processed_data:
            processed_data[problem_id] = {}
        if traj_extracted_answers not in processed_data[problem_id]:
            processed_data[problem_id][traj_extracted_answers] = {
                'anchored_responses': [],
            }
        processed_data[problem_id][traj_extracted_answers]['anchored_responses'].append(
            anchored_metadata
        )

    return processed_data


def compute_verification_stats(processed_data):
    counts = {"true": 0, "false": 0, "none": 0}
    for problem_entries in processed_data.values():
        for traj_extracted_answers, anchored_responses in problem_entries.items():
            for anchored_metadata in anchored_responses['anchored_responses']:
                verdict = anchored_metadata[1]["verdict"]
                if verdict is True:
                    counts["true"] += 1
                elif verdict is False:
                    counts["false"] += 1
                else:
                    counts["none"] += 1
    return counts


def load_init_responses(processed_data, init_response_path):
    """Load the init-response distribution for problems remaining in
    ``processed_data`` (anchored set) and attach as ``init_response`` lists."""
    with open(init_response_path, 'r') as f:
        init_response_data = [json.loads(line) for line in f]

    extracted_answers = []
    for entry in init_response_data:
        problem_id = entry['id']
        if problem_id not in processed_data:
            continue
        extracted_answers.append([
            problem_id,
            entry['init_response_generations_extracted_answer'],
            {
                "finish_reason": entry['init_response_generations_finish_reason'],
                "correctness": entry['init_response_generations_correctness'],
            },
        ])

    def parse_init_response_answer(item):
        problem_id, extracted_answer, metadata = item
        try:
            parsed_answer = parse_extracted_answer(extracted_answer)
        except Exception:
            parsed_answer = None
        return [problem_id, parsed_answer, metadata]

    print(f"Parsing {len(extracted_answers)} init-response answers")
    parsed_answers_ls = Parallel(n_jobs=-1)(
        delayed(parse_init_response_answer)(item) for item in extracted_answers
    )

    for problem_id, parsed_answer, metadata in parsed_answers_ls:
        if parsed_answer is None:
            continue
        assert problem_id in processed_data, f"Problem {problem_id} not in processed data"
        for traj_extracted_answers in processed_data[problem_id]:
            processed_data[problem_id][traj_extracted_answers].setdefault(
                "init_response", []
            ).append([parsed_answer, metadata])
    return processed_data
