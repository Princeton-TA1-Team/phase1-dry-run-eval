"""Evaluation driver for response-trajectory datasets.

The math verb routes to `is_equivalent_math`; the game_of_24 verb
(separate subpackage) calls `run_evaluation` directly with its own
verifier. The driver itself is verifier-agnostic — every step from
dataset load through visualization runs the same way regardless of
which verifier is in use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

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


def run_evaluation(args, *, verifier: Callable) -> int:
    """Run the dataset evaluation pipeline with an explicit verifier.

    Both `eval math` (verifier=is_equivalent_math) and `eval game_of_24`
    (verifier=is_correct_game_of_24) call into this function.
    """
    if args.data_format != "general_inference":
        assert args.problem_data_path_root is not None, (
            "Problem data path root is required for non-general inference data format"
        )

    dataset_dir = args.dataset_dir
    output_path = args.output
    n_jobs = args.n_jobs
    flatten_dataset = args.flatten_dataset
    response_column = args.response_column
    answer_column = args.answer_column

    if not Path(dataset_dir).exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        return 1

    print("Starting dataset evaluation...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output file: {output_path}")

    if args.data_format == "general_inference":
        if not args.single_partition:
            is_complete, missing_files = check_dataset_completeness(dataset_dir)
            if not is_complete:
                print("ERROR: Dataset is incomplete!")
                print("Missing or empty files:")
                for file in missing_files:
                    print(f"  - {file}")
                return 1
            print("SUCCESS: Dataset is complete!")

    if args.data_format != "general_inference":
        dataset = preprocess_api_data(args)
        dataset_dir = args.dataset_dir
    else:
        dataset = load_dataset(dataset_dir)
    print(f"SUCCESS: Loaded {len(dataset)} total entries")

    evaluated_dataset = evaluate_responses(dataset, answer_column, response_column, verifier, n_jobs=n_jobs)
    error_stats = analyze_response_errors(evaluated_dataset, answer_column, response_column, verifier)
    print_evaluation_summary(evaluated_dataset, error_stats, response_column)

    if output_path is None:
        dataset_name = Path(dataset_dir.rstrip("/")).name
        output_path = str(Path(dataset_dir) / f"evaluated_{dataset_name}.jsonl")

    save_evaluated_dataset(evaluated_dataset, output_path, flatten=flatten_dataset, response_column=response_column)
    save_error_analysis(error_stats, output_path)

    print("\nGenerating visualizations...")
    create_unified_correctness_plot(evaluated_dataset, output_path, answer_column, response_column)
    create_finish_reason_plot(error_stats, output_path, evaluated_dataset)

    print("\nSUCCESS: Evaluation completed successfully!")
    return 0


def main(args) -> int:
    """`eval math` entrypoint: verifier = is_equivalent_math."""
    return run_evaluation(args, verifier=is_equivalent_math)
