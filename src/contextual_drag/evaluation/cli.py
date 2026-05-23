"""Top-level CLI for contextual_drag.evaluation.

Per-cell verbs only: `math` and `crux` (ported verbatim from PR-2).
"""
from __future__ import annotations

import scriptconfig as scfg

from contextual_drag.evaluation.crux.eval_cli import EvalCruxCLI
from contextual_drag.evaluation.math.eval_cli import EvalMathCLI


class EvalCLI(scfg.ModalCLI):
    math = EvalMathCLI
    crux = EvalCruxCLI
