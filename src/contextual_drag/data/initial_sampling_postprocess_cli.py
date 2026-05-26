from __future__ import annotations

import scriptconfig as scfg


class InitialSamplingPostprocessCLI(scfg.DataConfig):
    __command__ = "initial-sampling-postprocess"

    input_dir = scfg.Value(None, required=True, help="Input directory containing flattened JSONL files.")
    input_file_template = scfg.Value("*/*/*flattened.jsonl", help="Glob pattern for input files.")
    max_response_length = scfg.Value(16384, type=int, help="Maximum retained response length.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import initial_sampling_postprocess

        return initial_sampling_postprocess.main(args)
