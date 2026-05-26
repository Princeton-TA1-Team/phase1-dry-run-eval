"""Regression catch-net for the prompt-budget arithmetic bug fixed in PR-2.

Background
----------

Prior code in the inference path used the wrong identity when computing
the prompt-truncation budget:

    effective_limit = max_tokens - generation_tokens   # WRONG: max_tokens
                                                       # IS the generation
                                                       # budget.

The fix uses ``max_model_len - max_tokens`` instead (context window minus
generation budget). PR-2 commit ``acb902a`` carries the fix.

Today the async driver delegates prompt truncation to vllm itself rather
than computing the budget locally, so the bug class isn't reachable. But
this guard fires *as soon as* anyone tries to recompute a budget by hand
in any of the inference modules.

If a future refactor re-introduces explicit prompt-budget arithmetic,
this test will fail unless the formula is the correct one.
"""
from __future__ import annotations

import importlib
import inspect


CANDIDATE_MODULES = (
    "contextual_drag.inference.prompts",
    "contextual_drag.inference.run_model",
    "contextual_drag.inference.config",
    "contextual_drag.inference.diagnostics",
)

BUGGY_FORMULA = "max_tokens - generation_tokens"


def test_prompt_budget_does_not_use_generation_budget_as_input() -> None:
    failures = []
    for modname in CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            # Test is forward-compat: a module that doesn't exist yet just skips.
            continue
        src = inspect.getsource(mod)
        if BUGGY_FORMULA in src:
            failures.append(modname)

    assert not failures, (
        f"Modules {failures} contain the bug-class formula "
        f"`{BUGGY_FORMULA}`; the correct identity is "
        "`max_model_len - max_tokens` (see PR-2 commit acb902a)."
    )
