from __future__ import annotations

import scriptconfig as scfg


class Stage1PostprocessIterativeCLI(scfg.DataConfig):
    __command__ = "stage1-postprocess-iterative"

    input_dir = scfg.Value(None, required=True, help="Input directory containing flattened JSONL files.")
    input_file_template = scfg.Value("*/*/*flattened.jsonl", help="Glob pattern for input files.")
    output_dir = scfg.Value(None, help="Optional output directory.")
    max_response_length = scfg.Value(16384, type=int, help="Maximum retained response length.")
    round_num = scfg.Value(0, type=int, help="Round number.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import stage1_postprocess_iterative

        return stage1_postprocess_iterative.main(args)
