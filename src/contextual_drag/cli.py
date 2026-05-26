from __future__ import annotations

import sys

import scriptconfig as scfg

from contextual_drag.analysis.cli import AnalysisCLI
from contextual_drag.data.cli import DataCLI
from contextual_drag.evaluation.cli import EvalCLI
from contextual_drag.inference.cli import InferenceCLI
from contextual_drag.mitigation.cli import MitigationCLI
from contextual_drag.recursive.cli import RecursiveCLI


class ContextualDragCLI(scfg.ModalCLI):
    inference = InferenceCLI
    eval = EvalCLI
    data = DataCLI
    mitigation = MitigationCLI
    recursive = RecursiveCLI
    analysis = AnalysisCLI


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    modal_argv = argv or ["--help"]
    result = ContextualDragCLI.main(argv=modal_argv, _noexit=True)
    if result == 1 and (not argv or "--help" in argv or "-h" in argv):
        return 0
    return int(result or 0)
