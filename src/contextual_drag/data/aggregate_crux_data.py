import math
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from contextual_drag.data.common import load_split_ids

def log_args(args):
    """
    Print out the arguments.
    """
    print(f"Arguments: {args}")
    print(f"Num True: {args.num_true}")
    print(f"Num False: {args.num_false}")
    print(f"Data Split: {args.data_split}")

def preprocess_dataset(ds, args):
    """
    Aggregate the dataset based on the given arguments.
    """
    problem_to_entries = {}

    print("Preprocessing dataset")
    print("Getting problem ids")
    problem_ids = list(ds[args.problem_id_column])
    print("Getting correctness")
    correctness_ls = list(ds[args.init_response_correctness_column])
    if args.filter_init_response_completeness:
        init_response_completeness_ls = list(ds['init_response_generations_finish_reason'])

    if args.filter_init_response_completeness:
        init_response_completeness_ls = list(ds['init_response_generations_finish_reason'])

    if args.filter_init_response_parsable_thinking:
        init_response_thinking_status_ls = list(ds['init_response_thinking_status'])
    
    print(f"Number of entries: {len(ds)}")
    for i in tqdm(range(len(problem_ids))):
        problem_id = problem_ids[i]
        if problem_id not in problem_to_entries:
            problem_to_entries[problem_id] = {"correct": [], "incorrect": [], "correct_sv": [], "incorrect_sv": []}
        
        correctness = correctness_ls[i]

        if args.filter_init_response_completeness:
            init_response_completeness = init_response_completeness_ls[i]
            if init_response_completeness != 'stop':
                continue
            
        if args.filter_init_response_parsable_thinking:
            init_response_thinking_status = init_response_thinking_status_ls[i]
            if init_response_thinking_status != 'parsable_thinking':
                continue

        if args.filter_init_response_completeness:
            init_response_completeness = init_response_completeness_ls[i]
            if init_response_completeness != 'stop':
                continue

        if correctness:
            problem_to_entries[problem_id]["correct"].append(i)
        else:
            problem_to_entries[problem_id]["incorrect"].append(i)
    
    return problem_to_entries

def combination(m, n):
    """
    Calculate the number of combinations of m choose n.
    """
    return math.comb(m, n)

def filter_problem(problem_to_entries, args):
    """
    Validate the problem to entry mapping.
    """
    total_valid_combos = 0
    correct_key = "correct"
    incorrect_key = "incorrect"
    valid_problem_ids = []
    for problem_id in problem_to_entries:
        num_trajs = args.num_true + args.num_false
        # if len(problem_to_entries[problem_id][correct_key]) >= num_trajs and len(problem_to_entries[problem_id][incorrect_key]) >= num_trajs:
        if len(problem_to_entries[problem_id][incorrect_key]) >= num_trajs:
            valid_problem_ids.append(problem_id)
            total_valid_combos += combination(len(problem_to_entries[problem_id][correct_key]), args.num_true) * combination(len(problem_to_entries[problem_id][incorrect_key]), args.num_false)
    print(f"Total valid combinations: {total_valid_combos}")
    return valid_problem_ids


def sample_combos(ds, problem_to_entries, sampled_problem_ids, args):
    """
    Sample the problems from the dataset.
    """
    sampled_problems = []
    correct_key = "correct"
    incorrect_key = "incorrect"

    n_correct = args.num_true
    n_incorrect = args.num_false

    def sample_combo(problem_id):
        correct_indices = np.random.choice(problem_to_entries[problem_id][correct_key], n_correct, replace=False)
        incorrect_indices = np.random.choice(problem_to_entries[problem_id][incorrect_key], n_incorrect, replace=False)
        indices = correct_indices.tolist() + incorrect_indices.tolist()
        np.random.shuffle(indices)

        key_ind = indices[0]

        # Basic Metadata for the problem
        new_entry = {
            "id": ds[key_ind]["id"],
            "problem": ds[key_ind]["problem"],
            "answer": ds[key_ind]["answer"],
            "source": ds[key_ind]["source"],
            "domain": ds[key_ind]["domain"],
            "label": ds[key_ind]["label"],
            "code": ds[key_ind]["code"],
            "llama8b_solve_rate": ds[key_ind]["llama8b_solve_rate"],
        }

        # Add the trajectories and their metadata
        for i, traj_ind in enumerate(indices):
            new_entry[f"traj{i+1}"] = ds[traj_ind]["init_response_final"][:32768]
            new_entry[f"traj{i+1}_correctness"] = ds[traj_ind][args.init_response_correctness_column]
            new_entry[f"traj{i+1}_metadata"] = ds[traj_ind]
        return new_entry

    sampled_problems = [sample_combo(problem_id) for problem_id in tqdm(sampled_problem_ids)]
    # sample_combo)(problem_id) for problem_id in sampled_problem_ids)
    return sampled_problems

def main(args):
    log_args(args)
    np.random.seed(args.seed)

    # Go through the input directory and load the dataset
    input_dir = Path(args.input_dir)
    assert input_dir.exists()
    assert (input_dir / "dataset_info.json").exists(), "Input directory is not a valid Hugging Face dataset"
    ds = load_from_disk(str(input_dir))

    if args.data_split != 'none':
        split_ids = load_split_ids(args.data_split, split_root=args.split_root, execution_mode=args.execution_mode)
        print(f"Sampling from {args.data_split} split")
        ds = ds.filter(lambda x: x['id'] in split_ids, num_proc=16)
        print(f"Filtered dataset to {len(ds)} samples")
    
    print(f"Sampling from {args.init_response_models}")
    ds = ds.filter(lambda x: x['init_response_generations_metadata']['model_config_alias'] in args.init_response_models, num_proc=16)
    print(f"Filtered dataset to {len(ds)} samples")

    if args.init_response_models:
        ds = ds.filter(lambda x: x['init_response_generations_metadata']['model_config_alias'] in args.init_response_models, num_proc=16)
    print(f"Filtered dataset to {len(ds)} samples")

    # Ensure that there is "id" column in the dataset
    assert args.problem_id_column in ds.column_names, "Dataset does not have an 'id' column"
    assert args.init_response_correctness_column in ds.column_names, "Dataset does not have an 'init_response_generations_correctness' column"

    problem_to_entries = preprocess_dataset(ds, args)
    print(f"Preprocessed {len(problem_to_entries)} problems")
    print(f"Average number of trajectories: {np.mean([len(problem_to_entries[problem_id]['correct'] + problem_to_entries[problem_id]['incorrect']) for problem_id in problem_to_entries])}")
    valid_problem_ids = filter_problem(problem_to_entries, args)
    print(f"Valid problems: {len(valid_problem_ids)}")

    n_samples = len(valid_problem_ids)
    sampled_problem_ids = valid_problem_ids

    aggregated_data = sample_combos(ds, problem_to_entries, sampled_problem_ids, args)
    output_ds = Dataset.from_list(aggregated_data)
    print(f"Sampled {len(aggregated_data)} combinations")

    if args.output_dir is None:
        # use the parent directory of the input directory
        args.output_dir = str(input_dir.parent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.filter_init_response_completeness:
        output_file_name = f"minimal_aggregated_data_T{args.num_true}_F{args.num_false}_complete.ds"
    else:
        output_file_name = f"minimal_aggregated_data_T{args.num_true}_F{args.num_false}.ds"
    output_file_path = output_dir / output_file_name
    output_ds.save_to_disk(str(output_file_path))

    print(f"Saved aggregated data to {output_file_path}")
    return 0
