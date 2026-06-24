"""Streaming, task-aware solve verification for the mitigation pipeline.

Dispatches per `task`: math/qa_mc -> boxed-answer math equivalence;
24-game -> `is_correct_game_of_24` (formula uses the given numbers & == 24);
crux-i/crux-o -> the crux code-execution verifier (run the function on the
predicted input, compare output). Math is the default for unknown tasks.

Each solve generation is annotated in place with two new keys:
    extracted_answer: contents of the LAST `\\boxed{...}` in the response, or None
    correctness:      True | False | None  (None = no boxed answer found, or
                      verifier crashed and we fell back to a string compare that
                      didn't match)

We dispatch verification through a `ProcessPoolExecutor` (NOT threads) because
sympy holds the GIL. The pool is created lazily on first use and shared across
all rows in a cell. Caller is responsible for `close()` after the cell ends.

Module-level helpers `extract_boxed_answer` and `is_equivalent_math` are
direct ports from `big_math_rl/verifiable_evaluation/math_eval/utils/math_utils.py`
so this file has no runtime dependency on the upstream tree.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from typing import Optional


_BOX_RE = re.compile(r"\\boxed\{((?:[^{}]|{[^{}]*})*)\}")


def extract_boxed_answer(response_text: str) -> Optional[str]:
    """Return the contents of the LAST `\\boxed{...}` in the response, or None."""
    if not response_text:
        return None
    matches = _BOX_RE.findall(response_text)
    if not matches:
        return None
    return matches[-1].strip()


def _strip_leading_zero(s: str) -> str:
    if s.startswith("0") and "." not in s and s != "0":
        return s.lstrip("0")
    return s


def is_equivalent_math(ans1: Optional[str], ans2: Optional[str]) -> Optional[bool]:
    """Math-verify-based equivalence with string-compare fallback.

    `ans1` is the extracted model answer; `ans2` is the ground truth. Returns
    True if equivalent, False otherwise, None if both are None.
    """
    if ans1 is None and ans2 is None:
        return True
    if ans1 is None or ans2 is None:
        return False
    if ans1.strip() == ans2.strip():
        return True
    if len(ans1) > 100 or len(ans2) > 100:
        return ans1.strip() == ans2.strip()
    try:
        from math_verify import parse, verify
        answer = parse(f"${ans1}$")
        expected = parse(f"${ans2}$")
        return bool(verify(expected, answer))
    except Exception:
        return ans1.strip() == ans2.strip()


def _verify_one(text: str, ground_truth: Optional[str]) -> tuple[Optional[str], Optional[bool]]:
    extracted = extract_boxed_answer(text)
    if extracted is None or ground_truth is None:
        return extracted, None
    return extracted, is_equivalent_math(extracted, ground_truth)


# Task families that need a non-math verifier (names match the `dataset` /
# `task_name` the card nodes pass through to the mitigation pipeline).
GAME24_TASKS = {"24-game"}
CRUX_TASKS = {"crux-i", "crux-o"}


def _verify_one_game24(text, ground_truth):
    """24-game: boxed formula must use the supplied numbers and evaluate to 24."""
    from contextual_drag.evaluation.game_of_24.verification import is_correct_game_of_24
    extracted = extract_boxed_answer(text)
    if extracted is None or ground_truth is None:
        return extracted, None
    return extracted, is_correct_game_of_24(extracted, ground_truth)


def _crux_cio_mode(row: dict):
    """Pull (code, input, output, mode) from a solve row (direct fields or the
    carried `traj1_metadata`), mirroring crux `evaluate_single_entry`."""
    info = row if ("mode" in row and "code" in row) else (row.get("traj1_metadata") or {})
    return (info.get("code"), info.get("input"), info.get("output"), info.get("mode"))


def _verify_one_crux(text, cio_mode):
    """crux: execute the function on the predicted input and compare output."""
    from contextual_drag.evaluation.crux.utils.evaluation_utils import evaluate_single_response
    code, inp, out, mode = cio_mode
    res = evaluate_single_response({"generated_response": text or ""}, (code, inp, out), mode)
    return res.get("extracted_answer"), res.get("correctness")


def _select_verifier(task, ground_truth, row):
    """Pick (worker, per-generation arg) for the task. Default: math."""
    t = (task or "").strip()
    if t in GAME24_TASKS:
        return _verify_one_game24, (row or {}).get("answer", ground_truth)
    if t in CRUX_TASKS:
        return _verify_one_crux, _crux_cio_mode(row or {})
    return _verify_one, ground_truth


class Verifier:
    """Async fan-out wrapper around a ProcessPoolExecutor.

    One Verifier per inference cell. Annotates a list of generation dicts in
    place with `extracted_answer` and `correctness`. Use as:

        verifier = Verifier(max_workers=8)
        await verifier.annotate(generations, ground_truth)
        ...
        verifier.close()
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._pool: concurrent.futures.ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._pool is None:
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers)
        return self._pool

    async def annotate(self, generations: list[dict],
                       ground_truth: Optional[str] = None, *,
                       task: Optional[str] = None,
                       row: Optional[dict] = None) -> list[dict]:
        if not generations:
            return generations
        pool = self._ensure_pool()
        loop = asyncio.get_running_loop()
        worker, arg = _select_verifier(task, ground_truth, row)
        futures = [
            loop.run_in_executor(pool, worker,
                                 g.get("generated_response", ""), arg)
            for g in generations
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        for g, res in zip(generations, results):
            if isinstance(res, Exception):
                # Verifier subprocess crashed on this sample. Treat as
                # "extracted answer unknown, correctness unknown" so the row
                # still lands on disk and resume isn't blocked.
                g["extracted_answer"] = None
                g["correctness"] = None
                continue
            extracted, ok = res
            g["extracted_answer"] = extracted
            g["correctness"] = ok
        return generations

    async def verify_pairs(self, pairs: list[tuple[str, Optional[str]]]
                           ) -> list[tuple[Optional[str], Optional[bool]]]:
        """Verify a flat list of (response_text, ground_truth) pairs.

        Unlike `annotate`, every pair has its own ground-truth — this is the
        common case for stage-0v (one trajectory per problem) where the
        verifier processes the whole dataset cross-row in one fan-out.

        Returns a list aligned with `pairs`. Each entry is (extracted_answer,
        correctness). On per-pair worker exceptions, returns (None, None) so
        the caller can still tabulate stats.
        """
        if not pairs:
            return []
        pool = self._ensure_pool()
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(pool, _verify_one, t, g)
                   for t, g in pairs]
        results = await asyncio.gather(*futures, return_exceptions=True)
        out: list[tuple[Optional[str], Optional[bool]]] = []
        for r in results:
            if isinstance(r, Exception):
                out.append((None, None))
            else:
                out.append(r)
        return out

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
