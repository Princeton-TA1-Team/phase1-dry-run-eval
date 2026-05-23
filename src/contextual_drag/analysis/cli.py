"""Umbrella CLI for contextual_drag.analysis; registers the three subgroups."""
from __future__ import annotations

import scriptconfig as scfg

from contextual_drag.analysis.error_conditioning.cli import ErrorConditioningCLI
from contextual_drag.analysis.mitigation_buckets.cli import MitigationBucketsCLI
from contextual_drag.analysis.ted.cli import TedCLI


class AnalysisCLI(scfg.ModalCLI):
    error_conditioning = ErrorConditioningCLI
    ted = TedCLI
    mitigation_buckets = MitigationBucketsCLI
