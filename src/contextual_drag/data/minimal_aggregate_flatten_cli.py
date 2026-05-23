from __future__ import annotations

import scriptconfig as scfg


class MinimalAggregateFlattenCLI(scfg.DataConfig):
    __command__ = "minimal-aggregate-flatten"

    input_ds_path = scfg.Value(None, required=True, help="Input dataset path.")
    output_ds_path = scfg.Value(None, help="Optional explicit output dataset path.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import minimal_aggregate_flatten

        return minimal_aggregate_flatten.main(args)
