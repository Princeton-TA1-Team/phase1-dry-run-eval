from __future__ import annotations

import scriptconfig as scfg


class EvalCruxCLI(scfg.DataConfig):
    __command__ = "crux"

    dataset_dir = scfg.Value(None, required=True, help="Directory containing evaluation dataset JSONL partitions.")
    single_partition = scfg.Value(False, isflag=True, help="Evaluate a single partition only.")
    output = scfg.Value(None, help="Optional output path.")
    n_jobs = scfg.Value(8, type=int, help="Number of parallel jobs.")
    flatten_dataset = scfg.Value(False, isflag=True, help="Save flattened evaluation output.")
    response_column = scfg.Value("init_response_generations", help="Response column name.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.evaluation.crux import eval as eval_impl

        return eval_impl.main(args)
