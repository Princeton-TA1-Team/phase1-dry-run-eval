"""
Evaluation utilities for response analysis and statistics.
"""

from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from .code_utils import extract_code_answer
from .crux_utils_general import evaluate_score
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def evaluate_single_response(response: Dict[str, Any], cio: Tuple[str, str, str], mode: str) -> Dict[str, Any]:
    """
    Evaluate a single response by comparing extracted answer with ground truth.
    This function is designed to be used in parallel processing.

    Args:
        response: The response dictionary
        ground_truth: The ground truth answer

    Returns:
        The response dictionary with added 'correctness' and 'extracted_answer' fields
    """
    response_text = response.get('generated_response', '')

    # Extract answer from response
    extracted_answer = extract_code_answer(response_text, mode)

    # print("-"*100)
    # print(cio[0])
    # print(cio[1], cio[2], extracted_answer)
    # print(response_text)
    # print('='*100)


    if extracted_answer is None:
        correctness = [False]
    else:
        correctness = evaluate_score([[extracted_answer], cio, mode])
        # print(correctness)

    # Add correctness to response (create a copy to avoid modifying the original)
    response_copy = response.copy()
    response_copy['correctness'] = correctness[0]
    response_copy['extracted_answer'] = extracted_answer

    return response_copy


def evaluate_single_entry(entry: Dict[str, Any], response_column: str) -> Dict[str, Any]:
    """
    Evaluate all responses for a single entry.
    This function is designed to be used in parallel processing.

    Args:
        entry: A single dataset entry with question and responses

    Returns:
        The entry with evaluated responses
    """

    info_entry = entry if 'mode' in entry and 'code' in entry else entry['traj1_metadata']
    code_val = info_entry.get('code')
    input_val = info_entry.get('input')
    output_val = info_entry.get('output')
    mode_val = info_entry.get('mode')

    response_column = response_column
    answer_column = "answer"

    cio = (code_val, input_val, output_val)

    if response_column not in entry:
        return entry

    # Create a copy of the entry to avoid modifying the original
    entry_copy = entry.copy()
    entry_copy[response_column] = []

    # Evaluate each response
    for response in entry[response_column]:
        evaluated_response = evaluate_single_response(response, cio, mode_val)
        entry_copy[response_column].append(evaluated_response)

    return entry_copy


def evaluate_responses(dataset: List[Dict[str, Any]], answer_column: str, response_column: str, n_jobs: int = None) -> List[Dict[str, Any]]:
    """
    Evaluate all responses in the dataset by comparing extracted answers with ground truth.
    Uses parallel processing for improved performance.
    Adds 'correctness' field to each response.

    Args:
        dataset: List of dataset entries
        n_jobs: Number of parallel jobs. If None, uses all available CPU cores.

    Returns:
        Dataset with evaluated responses
    """
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), len(dataset))

    print(f"Evaluating responses using {n_jobs} parallel jobs...")

    # evaluated_entries = [evaluate_single_entry(entry) for entry in tqdm(dataset, desc="Evaluating")]
    # Use joblib to parallelize the evaluation process with multiprocessing
    with ProcessPoolExecutor() as executor:
        # Pass response_column as a repeated iterable so evaluate_single_entry (a top-level function) is picklable
        evaluated_entries = executor.map(evaluate_single_entry, dataset, [response_column] * len(dataset))
    evaluated_entries = list(evaluated_entries)


    return evaluated_entries

def analyze_response_errors(dataset: List[Dict[str, Any]], answer_column: str, response_column: str) -> Dict[str, Any]:
    """
    Analyze incorrect and unparsable responses for detailed statistics.
    Includes pass@k metrics and sample incorrect answers.
    """
    # Calculate pass@k metrics by source
    pass_at_k_stats = pass_and_maj_at_k_by_source(dataset, answer_column, response_column)
    print(pass_at_k_stats)

    # Basic error statistics
    error_stats = {
        'pass_at_k_by_source': pass_at_k_stats,
        'unparsable_responses': [],
        'parsing_errors': [],
        'common_incorrect_patterns': {},
        'questions_with_all_incorrect': 0,
        'questions_with_all_unparsable': 0,
        'finish_reason_stats': {},
        'finish_reason_correctness': {}
    }
    if 'overall_stats' in pass_at_k_stats:
        error_stats['overall_stats'] = pass_at_k_stats['overall_stats']

    for entry in dataset:
        if response_column not in entry:
            print(f"Warning: {response_column} not found in entry {entry.get('id', 'unknown')}")
            continue

        question_id = entry.get('id', 'unknown')
        ground_truth = entry.get(answer_column, 'unknown')
        has_correct = False
        has_parsable = False
        all_unparsable = True

        for response in entry[response_column]:
            correctness = response.get('correctness')
            extracted_answer = response.get('extracted_answer')

            # Track finish reason statistics
            finish_reason = response.get('finish_reason', 'unknown')
            error_stats['finish_reason_stats'][finish_reason] = error_stats['finish_reason_stats'].get(finish_reason, 0) + 1

            # Track finish reason vs correctness correlation
            if finish_reason not in error_stats['finish_reason_correctness']:
                error_stats['finish_reason_correctness'][finish_reason] = {'correct': 0, 'incorrect': 0, 'unparsable': 0}

            if correctness is True:
                error_stats['finish_reason_correctness'][finish_reason]['correct'] += 1
                has_correct = True
                has_parsable = True
                all_unparsable = False
            elif correctness is False:
                error_stats['finish_reason_correctness'][finish_reason]['incorrect'] += 1
                has_parsable = True
                all_unparsable = False

                # Track only a few common incorrect patterns (limit to top 5)
                if extracted_answer and len(error_stats['common_incorrect_patterns']) < 5:
                    pattern = extracted_answer[:30]  # First 30 chars as pattern
                    error_stats['common_incorrect_patterns'][pattern] = error_stats['common_incorrect_patterns'].get(pattern, 0) + 1

            elif correctness is None:
                error_stats['finish_reason_correctness'][finish_reason]['unparsable'] += 1
                error_stats['unparsable_responses'].append({
                    'question_id': question_id,
                    'ground_truth': ground_truth,
                    'response_id': response.get('response_id'),
                    'response_text': response.get('generated_response', '')[:200]  # First 200 chars
                })

        # Check if all responses for this question are incorrect or unparsable
        if not has_correct and len(entry[response_column]) > 0:
            error_stats['questions_with_all_incorrect'] += 1
        if all_unparsable and len(entry[response_column]) > 0:
            error_stats['questions_with_all_unparsable'] += 1

    # Keep only top 5 most common incorrect patterns
    if error_stats['common_incorrect_patterns']:
        sorted_patterns = sorted(error_stats['common_incorrect_patterns'].items(),
                               key=lambda x: x[1], reverse=True)
        error_stats['common_incorrect_patterns'] = dict(sorted_patterns[:5])

    return error_stats

def get_majority_correctness_and_pass_at_k(answer_ls, correctness_ls):

    # If there exist a true in correctness_ls, return True
    pass_at_k = True if True in correctness_ls else False

    n_correct = 0
    answer_to_correctness = {}
    for answer, correctness in zip(answer_ls, correctness_ls):
        if correctness == True:
            n_correct += 1
        answer_to_correctness[answer] = correctness

    # We know majority vote is correct if more than half of the answers are correct
    if n_correct > len(answer_ls) / 2:
        return True, True

    if n_correct == 0:
        return False, False

    answer_bins = {}
    for answer in answer_ls:
        is_unique = True
        for answer_key in answer_bins:
            if answer.strip() == answer_key.strip():
                answer_bins[answer_key] += 1
                is_unique = False
                break
        if is_unique:
            answer_bins[answer] = 1
    max_freq = max(answer_bins.values())
    max_freq_correctness = []
    for answer_key in answer_bins:
        if answer_bins[answer_key] == max_freq:
            max_freq_correctness.append(answer_to_correctness[answer_key])

    return bool(np.random.choice(max_freq_correctness)), pass_at_k

def compute_majk_and_passk_sliding(answer_ls, correctness_ls, k):
    assert k <= len(answer_ls), "k must be less than or equal to the length of answer_ls"
    assert len(answer_ls) == len(correctness_ls), "answer_ls and correctness_ls must have the same length"
    pass_at_k_ls = []
    maj_at_k_ls = []

    answer_ls_dup = answer_ls * 2
    correctness_ls_dup = correctness_ls * 2
    for i in range(len(answer_ls)):
        # Get a sliding window of size k that starts at index i and wraps around
        answer_window = answer_ls_dup[i:i+k]
        correctness_window = correctness_ls_dup[i:i+k]
        maj_correctness, pass_at_k = get_majority_correctness_and_pass_at_k(answer_window, correctness_window)
        maj_at_k_ls.append(maj_correctness)
        pass_at_k_ls.append(pass_at_k)

    return maj_at_k_ls, pass_at_k_ls

def compute_majk_and_passk_stats(answer_ls, correctness_ls):
    pass_at_k_stats = []
    maj_at_k_stats = []
    for k in range(1, len(answer_ls) + 1):
        maj_at_k_ls, pass_at_k_ls = compute_majk_and_passk_sliding(answer_ls, correctness_ls, k)
        pass_at_k_stats.append(pass_at_k_ls)
        maj_at_k_stats.append(maj_at_k_ls)
    maj_at_k_stats = np.array(maj_at_k_stats)
    pass_at_k_stats = np.array(pass_at_k_stats)
    return maj_at_k_stats, pass_at_k_stats

def pass_and_maj_at_k_by_source(dataset: List[Dict[str, Any]], answer_column: str, response_column: str) -> Dict[str, Dict[str, Any]]:
    """
    Calculate pass@k metrics for each data source.

    Returns:
        Dictionary mapping source -> {
            'pass_at_k': [pass@1, pass@2, ..., pass@n],
            'total_questions': int,
            'n_trajectories': int,
            'overall_correctness': float
        }
    """
    print("Calculating pass@k and maj@k metrics by source...")

    source_to_inds = {'overall': list(range(len(dataset)))}

    extracted_answer_correctness_ls = []

    for i, entry in enumerate(dataset):
        source = entry.get('source', 'unknown')
        if source not in source_to_inds:
            source_to_inds[source] = []
        source_to_inds[source].append(i)

        answer_ls = []
        correctness_ls = []

        for response in entry.get(response_column, []):
            answer_ls.append(response.get('extracted_answer', 'unknown'))
            correctness_ls.append(response.get('correctness', 'unknown'))
        extracted_answer_correctness_ls.append((answer_ls, correctness_ls))

    # For each k, compute pass@k and maj@k
    # Compute pass@k and maj@k for every entry in parallel

    stats_ls = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(compute_majk_and_passk_stats)(answer_ls, correctness_ls)
        for answer_ls, correctness_ls in tqdm(extracted_answer_correctness_ls, desc="Computing pass@k and maj@k stats")
    )

    maj_at_k_stats = np.array([stats[0] for stats in stats_ls])
    pass_at_k_stats = np.array([stats[1] for stats in stats_ls])

    # Compute ov
    results = {}
    for source in source_to_inds:
        maj_at_k_stats_source = maj_at_k_stats[source_to_inds[source]]
        pass_at_k_stats_source = pass_at_k_stats[source_to_inds[source]]

        mean_maj_at_k = np.mean(maj_at_k_stats_source, axis=0)
        mean_pass_at_k = np.mean(pass_at_k_stats_source, axis=0)

        std_maj_at_k = np.std(mean_maj_at_k, axis=-1)
        std_pass_at_k = np.std(mean_pass_at_k, axis=-1)

        supermean_maj_at_k = np.mean(mean_maj_at_k, axis=-1)
        supermean_pass_at_k = np.mean(mean_pass_at_k, axis=-1)

        results[source] = {}
        results[source]['maj_at_k'] = supermean_maj_at_k.tolist()
        results[source]['pass_at_k'] = supermean_pass_at_k.tolist()
        results[source]['std_maj_at_k'] = std_maj_at_k.tolist()
        results[source]['std_pass_at_k'] = std_pass_at_k.tolist()

        results[source]['total_questions'] = len(maj_at_k_stats_source)
        results[source]['n_trajectories'] = len(maj_at_k_stats_source[0])
        results[source]['overall_correctness'] = supermean_maj_at_k[0]

    return results
