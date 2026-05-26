"""Session-wide stubs so non-GPU pytest runs never load real vllm.

The smoke test in ``test_smoke.py::test_group_help_runs_without_heavy_imports``
already proves that the CLI's ``--help`` path is import-light. This
conftest generalises that: it patches ``sys.modules['vllm']`` (and
``transformers``, since the inference module top-level imports both)
before any test module is imported, so test files that need to inspect
``contextual_drag.inference.run_model`` etc. can collect without
dragging in CUDA libraries.

Design notes
------------
Attribute access on a stubbed module returns a *real Python class*
(constructed via a metaclass), not an instance. That matters for two
reasons:

  1. ``from transformers import AutoTokenizer`` binds a usable name.
  2. ``isinstance``/``issubclass`` guards in third-party libraries
     (notably ``datasets`` fingerprinting, which calls
     ``issubclass(obj_type, transformers.PreTrainedTokenizerBase)``)
     accept the stub without raising ``TypeError``.

The class is harmless until someone **calls** it, at which point the
metaclass raises the canonical RuntimeError. That's our "non-trivial
attribute access" boundary: import + isinstance pass through; actual
invocation fails loudly with a message pointing at this file.
"""
from __future__ import annotations

import sys
import types


_STUBBED = ("vllm", "transformers")


class _StubMeta(type):
    """Metaclass: attribute access yields another stub class; calling raises."""

    def __getattr__(cls, item: str):  # noqa: D401
        if item.startswith("__"):
            raise AttributeError(item)
        return _make_stub_class(f"{cls.__name__}.{item}")

    def __call__(cls, *args, **kwargs):
        raise RuntimeError(
            f"{cls.__name__} called during a non-GPU test "
            "(vllm/transformers are stubbed by tests/conftest.py); "
            "mark @pytest.mark.gpu to opt in."
        )


def _make_stub_class(name: str) -> type:
    """Forge a real Python class named *name* via the stub metaclass."""
    return _StubMeta(name, (), {})


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make_stub_class(f"{self.__name__}.{name}")


def _install_stub(name: str) -> None:
    existing = sys.modules.get(name)
    if isinstance(existing, _StubModule):
        return
    sys.modules[name] = _StubModule(name)


def pytest_configure(config) -> None:  # noqa: D401
    """Install the stubs as early as pytest collection allows."""
    del config  # unused
    for name in _STUBBED:
        _install_stub(name)
