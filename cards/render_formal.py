"""Render full-dataset "formal" evaluation cards for a model.

One card per (model, test, dataset) at
``cards/formal_test/<model>/<test>/<dataset>.yaml``. Each test has a fixed
dataset set + default config; for a new model we just re-render. Recursive tests
are intentionally out of scope (restricted to GPT_OSS_20B, hand-authored).

Design: each test's committed smoke card under ``cards/smoke_runs/`` is used as
the *prototype* — we deep-copy it (claim, symbols, pipeline node, structure
verbatim, so the rendered card satisfies the same contract as the smoke cards)
and patch in the target model, the full-data ``data_path``, the dataset-family
template keys (via ``cards.nodes._dataset_registry``), and the formal run sizing.

    python -m cards.render_formal --model Qwen3_8B_NoThinking
    python -m cards.render_formal --model GPT_OSS_20B --test drag mitigation
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import scriptconfig as scfg
import yaml

from cards.nodes._dataset_registry import (
    data_path_for,
    framing_template_key_for,
    init_template_key_for,
    onef_template_key_for,
    twof_template_key_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Standard 8-benchmark set for drag / error-conditioning / mitigation.
_EIGHT = ["aime24", "aime25", "hmmt24", "hmmt25", "gpqa", "mmlu", "crux-i", "24-game"]


def _patch_drag(ap: dict, ds: str) -> None:
    ap["init_template_key"] = init_template_key_for(ds)
    ap["twof_template_key"] = twof_template_key_for(ds)


def _patch_drag_1f(ap: dict, ds: str) -> None:
    ap["init_template_key"] = init_template_key_for(ds)
    ap["cond_template_key_1f"] = onef_template_key_for(ds)


def _patch_ec_posthoc(ap: dict, ds: str) -> None:
    # pure-analysis node: only model_config/data_path/dataset/cond_cache_root; nothing to patch.
    pass


def _patch_ec_external(ap: dict, ds: str) -> None:
    ap["regime"] = "framing"
    ap["init_template_key"] = init_template_key_for(ds)
    ap["framing_template_key"] = framing_template_key_for(ds)
    # keep the (unused-under-framing) 1f key family-correct in case regime is flipped
    ap["cond_template_key_1f"] = onef_template_key_for(ds)


def _patch_mitigation(ap: dict, ds: str) -> None:
    ap["init_template_key"] = init_template_key_for(ds)
    ap["onef_template_key"] = onef_template_key_for(ds)


def _patch_ted(ap: dict, ds: str) -> None:
    ap["init_template_key"] = init_template_key_for(ds)
    ap["twof_template_key"] = twof_template_key_for(ds)


@dataclass(frozen=True)
class TestSpec:
    base: str                              # prototype smoke card, relative to cards/
    datasets: List[str]
    patch: Callable[[dict, str], None]
    has_dataset: bool                      # node carries a `dataset` algo_param (TED does not)
    sizing: bool = True                    # node takes n/max_tokens/max_questions (analysis nodes do not)


TESTS = {
    "drag": TestSpec(
        "smoke_runs/Qwen3_8B_NoThinking/drag/gpqa.yaml",
        _EIGHT, _patch_drag, True),
    "drag-1f": TestSpec(
        "smoke_runs/Qwen3_8B_NoThinking/drag-1f/gpqa.yaml",
        _EIGHT, _patch_drag_1f, True),
    "error-conditioning-posthoc": TestSpec(
        "smoke_runs/Qwen3_8B_NoThinking/error-conditioning-posthoc/aime24.yaml",
        _EIGHT, _patch_ec_posthoc, True, sizing=False),
    "error-conditioning-external": TestSpec(
        "smoke_runs/Qwen3_8B_NoThinking/error-conditioning-external/aime24.yaml",
        _EIGHT, _patch_ec_external, True),
    "mitigation": TestSpec(
        "smoke_runs/Qwen3_8B_NoThinking/mitigation/gpqa.yaml",
        _EIGHT, _patch_mitigation, True),
    "ted": TestSpec(
        "smoke_runs/Qwen3_8B_NoThinking/ted/24-game.yaml",
        ["24-game"], _patch_ted, False),
}


def _yaml_dump(obj) -> str:
    class _D(yaml.SafeDumper):
        pass

    def _str_rep(dumper, data):
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    _D.add_representer(str, _str_rep)
    return yaml.dump(obj, Dumper=_D, sort_keys=False, default_flow_style=False,
                     allow_unicode=True, width=100)


def _full_rows(repo_root: Path, dataset: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    p = repo_root / "data" / "full_data" / dataset / f"{dataset}.ds"
    try:
        from datasets import load_from_disk
        return int(len(load_from_disk(str(p))))
    except Exception as e:  # data not present at render time
        print(f"[render] WARN: could not read row count for {dataset} "
              f"({type(e).__name__}: {e}); using 100000.")
        return 100000


def _model_max_tokens(repo_root: Path, model: str, override: Optional[int]) -> int:
    """Per-model generation budget from eval_models_params.json (paper Table 2);
    falls back to context_length, then 32768. An explicit --max_tokens wins."""
    if override is not None:
        return int(override)
    import glob
    hits = glob.glob(str(repo_root / "src" / "**" / "eval_models_params.json"), recursive=True)
    if hits:
        try:
            params = (yaml.safe_load(Path(hits[0]).read_text()) or {}).get(model, {})
            mt = (params.get("sampling_params") or {}).get("max_tokens") or params.get("context_length")
            if mt:
                return int(mt)
        except Exception as e:  # malformed / model missing
            print(f"[render] WARN: could not read max_tokens for {model} "
                  f"({type(e).__name__}: {e}); using 32768.")
    else:
        print(f"[render] WARN: eval_models_params.json not found; max_tokens=32768 for {model}.")
    return 32768


def _render_one(*, test: str, dataset: str, model: str, repo_root: Path,
                n: int, max_tokens: int, max_questions: Optional[int]) -> dict:
    spec = TESTS[test]
    base_path = repo_root / "cards" / spec.base
    card = copy.deepcopy(yaml.safe_load(base_path.read_text()))

    card["title"] = f"Contextual Drag [{test}] — {model} × {dataset} (formal, full dataset)"
    card["description"] = (
        f"Formal full-dataset {test} run for model `{model}` on `{dataset}`.\n"
        f"Auto-generated by `python -m cards.render_formal`; claim/threshold are "
        f"inherited from a contextual-drag smoke-card prototype.\n"
    )

    node_key = next(iter(card["pipeline"]))
    ap = card["pipeline"][node_key]["algo_params"]
    ap["model_config"] = model
    ap["data_path"] = data_path_for(dataset)
    if spec.sizing:
        ap["max_questions"] = _full_rows(repo_root, dataset, max_questions)
        ap["n"] = n
        ap["max_tokens"] = max_tokens
    if spec.has_dataset:
        ap["dataset"] = dataset
    spec.patch(ap, dataset)
    return card


class RenderFormalCLI(scfg.DataConfig):
    __command__ = "render_formal"

    model = scfg.Value(None, required=True, help="Model alias from eval_models_params.json.")
    test = scfg.Value(None, nargs="+", choices=sorted(TESTS),
                      help="Tests to render (default: all five).")
    datasets = scfg.Value(None, nargs="+",
                          help="Override the dataset set for the selected test(s).")
    out_dir = scfg.Value("cards/formal_test",
                         help="Output root (relative to repo root unless absolute).")
    n = scfg.Value(16, type=int, help="Samples per question.")
    max_tokens = scfg.Value(
        None, type=int,
        help="Generation budget; default = the model's configured max_tokens in "
             "eval_models_params.json (paper Table 2: e.g. 32768 thinking, 65536 Nemotron).")
    max_questions = scfg.Value(None, type=int,
                               help="Cap rows; default = full dataset (row count read from disk).")
    force = scfg.Value(False, isflag=True, help="(Always overwrites existing cards.)")

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True)
        repo_root = REPO_ROOT
        out_root = Path(cfg.out_dir)
        if not out_root.is_absolute():
            out_root = repo_root / out_root

        tests = list(cfg.test) if cfg.test else sorted(TESTS)
        mq = None if cfg.max_questions is None else int(cfg.max_questions)
        resolved_mt = _model_max_tokens(repo_root, str(cfg.model), cfg.max_tokens)
        written = []
        for test in tests:
            if test not in TESTS:
                raise SystemExit(f"unknown test {test!r}; choices: {sorted(TESTS)}")
            datasets = list(cfg.datasets) if cfg.datasets else TESTS[test].datasets
            for ds in datasets:
                card = _render_one(
                    test=test, dataset=ds, model=str(cfg.model), repo_root=repo_root,
                    n=int(cfg.n), max_tokens=resolved_mt, max_questions=mq)
                dest = out_root / str(cfg.model) / test / f"{ds}.yaml"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(_yaml_dump(card))
                written.append(dest)
                try:
                    shown = dest.relative_to(repo_root)
                except ValueError:
                    shown = dest
                print(f"[render] wrote {shown}")
        print(f"[render] {len(written)} cards for model={cfg.model}, tests={tests}")
        return 0


def main(argv=None):
    return RenderFormalCLI.main(argv=argv)


if __name__ == "__main__":
    main()
