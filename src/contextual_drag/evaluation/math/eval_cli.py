from __future__ import annotations

import scriptconfig as scfg


class EvalMathCLI(scfg.DataConfig):
    __command__ = "math"

    dataset_dir = scfg.Value(None, required=True, help="Directory containing evaluation dataset JSONL partitions.")
    single_partition = scfg.Value(False, isflag=True, help="Evaluate a single partition only.")
    output = scfg.Value(None, help="Optional output path.")
    n_jobs = scfg.Value(8, type=int, help="Number of parallel jobs.")
    flatten_dataset = scfg.Value(False, isflag=True, help="Save flattened evaluation output.")
    answer_column = scfg.Value("answer", help="Answer column name.")
    response_column = scfg.Value("init_response_generations", help="Response column name.")
    equivalent_parser = scfg.Value("math_verify", help="Verification backend.")
    data_format = scfg.Value(
        "general_inference",
        choices=["general_inference", "openai_api", "gemini_api"],
        help="Input data format.",
    )
    problem_data_path_root = scfg.Value(None, help="Problem dataset path template for API-backed evaluation.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.evaluation.math import eval as eval_impl

        return eval_impl.main(args)
