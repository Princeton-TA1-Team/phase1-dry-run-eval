from __future__ import annotations

import scriptconfig as scfg


class AggregateDataIterativeCLI(scfg.DataConfig):
    __command__ = "aggregate-iterative"

    input_dir = scfg.Value("processed_flattened_outputs", help="Input dataset directory.")
    num = scfg.Value(0, type=int, help="Number of trajectories per sample.")
    output_dir = scfg.Value(None, help="Output directory.")
    seed = scfg.Value(42, type=int, help="Random seed.")
    problem_id_column = scfg.Value("id", help="Problem id column.")
    filter_init_response_completeness = scfg.Value(False, isflag=True, help="Require stop finish reason.")
    filter_init_response_parsable_thinking = scfg.Value(False, isflag=True, help="Require parsable thinking.")
    data_split = scfg.Value("none", help="Data split selector.")
    init_response_models = scfg.Value(
        ["Qwen3_8B_Thinking", "Qwen3_8B_NoThinking", "LlamaR1_8B", "Gemma3_4B", "Llama3.1_8B", "QwenR1_7B"],
        nargs="+",
        help="Model aliases to sample from.",
    )
    round_num = scfg.Value(0, type=int, help="Round number for recursive aggregation.")
    split_root = scfg.Value(None, help="Directory containing split JSON files.")
    execution_mode = scfg.Value(None, choices=["workspace", "installed"], help="Explicit execution mode.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import aggregate_data_iterative

        return aggregate_data_iterative.main(args)
