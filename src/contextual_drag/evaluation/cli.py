"""Top-level CLI for contextual_drag.evaluation.

Per-cell verbs: `math` (SymPy-based math-equivalence verifier),
`crux` (code-execution verifier), `game_of_24` (24-game arithmetic
verifier — equivalent to 24 AND digits = permutation of inputs).
"""
from __future__ import annotations

import scriptconfig as scfg

from contextual_drag.evaluation.crux.eval_cli import EvalCruxCLI
from contextual_drag.evaluation.game_of_24.eval_cli import EvalGameOf24CLI
from contextual_drag.evaluation.math.eval_cli import EvalMathCLI


class EvalCLI(scfg.ModalCLI):
    math = EvalMathCLI
    crux = EvalCruxCLI
    game_of_24 = EvalGameOf24CLI
