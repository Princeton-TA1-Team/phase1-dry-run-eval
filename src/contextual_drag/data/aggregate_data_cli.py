from __future__ import annotations

import scriptconfig as scfg


class AggregateDataCLI(scfg.DataConfig):
    __command__ = "aggregate"

    input_dir = scfg.Value("processed_flattened_outputs", help="Input dataset directory.")
    num_true = scfg.Value(0, type=int, help="Number of correct trajectories per sample.")
    num_false = scfg.Value(2, type=int, help="Number of incorrect trajectories per sample.")
    min_num_true_sampling = scfg.Value(2, type=int, help="Filter threshold: minimum number of correct responses a problem must have to be kept. Must be at least num_true.")
    min_num_false_sampling = scfg.Value(2, type=int, help="Filter threshold: minimum number of incorrect responses a problem must have to be kept. Must be at least num_false.")
    output_dir = scfg.Value(None, help="Output directory.")
    seed = scfg.Value(42, type=int, help="Random seed.")
    problem_id_column = scfg.Value("id", help="Problem id column.")
    init_response_correctness_column = scfg.Value(
        "init_response_generations_correctness", help="Correctness column name."
    )
    filter_init_response_completeness = scfg.Value(False, isflag=True, help="Require stop finish reason.")
    filter_init_response_parsable_thinking = scfg.Value(False, isflag=True, help="Require parsable thinking.")
    init_response_models = scfg.Value(
        ["Qwen3_8B_Thinking", "Qwen3_8B_NoThinking", "LlamaR1_8B", "Gemma3_4B", "Llama3.1_8B", "QwenR1_7B"],
        nargs="+",
        help="Model aliases to sample from.",
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import aggregate_data

        return aggregate_data.main(args)
