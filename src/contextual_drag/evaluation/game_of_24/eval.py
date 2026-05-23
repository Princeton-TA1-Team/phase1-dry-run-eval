"""`eval game_of_24` entrypoint: reuses the math eval driver with the
game-of-24 verifier."""
from __future__ import annotations

from contextual_drag.evaluation.game_of_24.verification import is_correct_game_of_24
from contextual_drag.evaluation.math.eval import run_evaluation


def main(args) -> int:
    return run_evaluation(args, verifier=is_correct_game_of_24)
