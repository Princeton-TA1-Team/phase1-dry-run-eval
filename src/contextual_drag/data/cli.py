"""Top-level CLI for contextual_drag.data (PR-2 verbs, hoisted from cli.py)."""
from __future__ import annotations

import scriptconfig as scfg

from contextual_drag.data.aggregate_crux_data_cli import AggregateCruxDataCLI
from contextual_drag.data.aggregate_data_cli import AggregateDataCLI
from contextual_drag.data.aggregate_data_iterative_cli import AggregateDataIterativeCLI
from contextual_drag.data.initial_sampling_postprocess_cli import InitialSamplingPostprocessCLI
from contextual_drag.data.minimal_aggregate_flatten_cli import MinimalAggregateFlattenCLI
from contextual_drag.data.stage1_postprocess_iterative_cli import Stage1PostprocessIterativeCLI


class DataCLI(scfg.ModalCLI):
    initial_sampling_postprocess = InitialSamplingPostprocessCLI
    minimal_aggregate_flatten = MinimalAggregateFlattenCLI
    aggregate = AggregateDataCLI
    aggregate_crux = AggregateCruxDataCLI
    aggregate_iterative = AggregateDataIterativeCLI
    stage1_postprocess_iterative = Stage1PostprocessIterativeCLI
