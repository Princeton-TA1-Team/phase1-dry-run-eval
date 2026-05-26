"""CLI surface for `eval game_of_24`.

Same fields as `EvalMathCLI` minus `--equivalent_parser` (the verifier is
fixed to game_of_24).
"""
from __future__ import annotations

import scriptconfig as scfg


class EvalGameOf24CLI(scfg.DataConfig):
    __command__ = "game_of_24"

    dataset_dir = scfg.Value(None, required=True, help="Directory containing evaluation dataset JSONL partitions.")
    single_partition = scfg.Value(False, isflag=True, help="Evaluate a single partition only.")
    output = scfg.Value(None, help="Optional output path.")
    n_jobs = scfg.Value(8, type=int, help="Number of parallel jobs.")
    flatten_dataset = scfg.Value(False, isflag=True, help="Save flattened evaluation output.")
    answer_column = scfg.Value("answer", help="Answer column name.")
    response_column = scfg.Value("init_response_generations", help="Response column name.")
    data_format = scfg.Value(
        "general_inference",
        choices=["general_inference", "openai_api", "gemini_api"],
        help="Input data format.",
    )
    problem_data_path_root = scfg.Value(None, help="Problem dataset path template for API-backed evaluation.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.evaluation.game_of_24 import eval as eval_impl

        return eval_impl.main(args)
