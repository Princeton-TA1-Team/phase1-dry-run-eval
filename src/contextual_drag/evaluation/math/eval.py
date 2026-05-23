#!/usr/bin/env python3
"""Evaluation script for math datasets with multiple response trajectories."""

from pathlib import Path

from contextual_drag.evaluation.math.utils.dataset_utils import check_dataset_completeness, load_dataset
from contextual_drag.evaluation.math.utils.evaluation_utils import evaluate_responses, analyze_response_errors
from contextual_drag.evaluation.math.utils.output_utils import (
    print_evaluation_summary,
    save_evaluated_dataset,
    save_error_analysis,
)
from contextual_drag.evaluation.math.utils.visualization_utils import (
    create_unified_correctness_plot,
    create_finish_reason_plot,
)
from contextual_drag.evaluation.math.utils.api_preprocessing import preprocess_api_data
from contextual_drag.evaluation.math.utils.math_utils import is_equivalent_math
from contextual_drag.evaluation.math.utils.verification_utils import is_correct_game_of_24

VERIFICATION_MAPPING = {
    "math_verify": is_equivalent_math,
    "game_of_24": is_correct_game_of_24
}


def verification_method_router(verification_method):
    if verification_method not in VERIFICATION_MAPPING:
        raise ValueError(f"Invalid verification method: {verification_method}, please choose from math_verify or game_of_24")
    return VERIFICATION_MAPPING[verification_method]


def main(args):
    if args.data_format != 'general_inference':
        assert args.problem_data_path_root is not None, "Problem data path root is required for non-general inference data format"
    
    # Validate dataset directory
    dataset_dir = args.dataset_dir
    output_path = args.output
    n_jobs = args.n_jobs
    flatten_dataset = args.flatten_dataset
    response_column = args.response_column
    answer_column = args.answer_column
    equivalent_parser_name = args.equivalent_parser
    print(f"Using equivalent parser: {equivalent_parser_name}")

    equivalent_parser = verification_method_router(equivalent_parser_name)

    if not Path(dataset_dir).exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        return 1
    
    
    print("Starting dataset evaluation...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output file: {output_path}")
    
    # Step 1: Check dataset completeness (only for multiple partitions)
    if args.data_format == 'general_inference':
        if not args.single_partition:
            is_complete, missing_files = check_dataset_completeness(dataset_dir)
            
            if not is_complete:
                print("ERROR: Dataset is incomplete!")
                print("Missing or empty files:")
                for file in missing_files:
                    print(f"  - {file}")
                return 1
            else:
                print("SUCCESS: Dataset is complete!")

    # Step 1.5: Preprocess API data if necessary
    if args.data_format != 'general_inference':
        dataset = preprocess_api_data(args)
        dataset_dir = args.dataset_dir

    # Step 2: Load and combine all partitions if necessary
    if args.data_format == 'general_inference':
        dataset = load_dataset(dataset_dir)
    print(f"SUCCESS: Loaded {len(dataset)} total entries")
    
    # Step 3: Evaluate responses
    evaluated_dataset = evaluate_responses(dataset, answer_column, response_column, equivalent_parser, n_jobs=n_jobs)
    
    # Step 4: Print summary and get error stats
    error_stats = analyze_response_errors(evaluated_dataset, answer_column, response_column, equivalent_parser)
    print_evaluation_summary(evaluated_dataset, error_stats, response_column)
    
    # Step 5: Save evaluated dataset and error analysis
    # Set default output path if not provided
    if output_path is None:
        # Extract the last part of the dataset directory name
        dataset_name = Path(dataset_dir.rstrip("/")).name
        output_path = str(Path(dataset_dir) / f"evaluated_{dataset_name}.jsonl")
    
    save_evaluated_dataset(evaluated_dataset, output_path, flatten=flatten_dataset, response_column=response_column)
    save_error_analysis(error_stats, output_path)
    
    # Step 6: Generate visualizations
    print("\nGenerating visualizations...")
    create_unified_correctness_plot(evaluated_dataset, output_path, answer_column, response_column)
    create_finish_reason_plot(error_stats, output_path, evaluated_dataset)
    
    print("\nSUCCESS: Evaluation completed successfully!")
    return 0
