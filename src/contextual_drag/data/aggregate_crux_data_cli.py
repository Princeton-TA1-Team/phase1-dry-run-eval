from __future__ import annotations

import scriptconfig as scfg


class AggregateCruxDataCLI(scfg.DataConfig):
    __command__ = "aggregate-crux"

    input_dir = scfg.Value("processed_flattened_outputs", help="Input dataset directory.")
    num_true = scfg.Value(0, type=int, help="Number of correct trajectories per sample.")
    num_false = scfg.Value(2, type=int, help="Number of incorrect trajectories per sample.")
    output_dir = scfg.Value(None, help="Output directory.")
    seed = scfg.Value(42, type=int, help="Random seed.")
    problem_id_column = scfg.Value("id", help="Problem id column.")
    init_response_correctness_column = scfg.Value(
        "init_response_generations_correctness", help="Correctness column name."
    )
    filter_init_response_parsable_thinking = scfg.Value(False, isflag=True, help="Require parsable thinking.")
    data_split = scfg.Value("none", help="Data split selector.")
    init_response_models = scfg.Value(None, nargs="+", help="Model aliases to sample from.")
    filter_init_response_completeness = scfg.Value(False, isflag=True, help="Require stop finish reason.")
    split_root = scfg.Value(None, help="Directory containing split JSON files.")
    execution_mode = scfg.Value(None, choices=["workspace", "installed"], help="Explicit execution mode.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import aggregate_crux_data

        return aggregate_crux_data.main(args)
