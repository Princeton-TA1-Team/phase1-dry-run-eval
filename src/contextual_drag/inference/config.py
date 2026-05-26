"""Model configuration loading + vLLM sampling-params construction.

Reads `eval_models_params.json` (the packaged resource shipped under
`contextual_drag.resources.inference`) and produces a `vllm.SamplingParams`
from the per-model sampling block, with optional CLI overrides for
`max_tokens`. Pass `config_path=None` to use the packaged resource.
"""

import json
from typing import Any, Optional

from vllm import SamplingParams

from contextual_drag.config.resources import inference_model_config_resource


def load_model_config(config_path: Optional[str], alias: str) -> dict:
    """Load a single model's config block by alias. None ⇒ packaged resource."""
    if config_path is None:
        configs = inference_model_config_resource()
        source = "<packaged contextual_drag.resources.inference.eval_models_params.json>"
    else:
        with open(config_path) as f:
            configs = json.load(f)
        source = config_path
    if alias not in configs:
        raise SystemExit(
            f"Model alias {alias!r} not found in {source}. "
            f"Available: {sorted(configs)}"
        )
    return configs[alias]


def make_sampling_params(model_config: dict, n_samples: int, seed: int,
                         max_tokens_override: Optional[int] = None) -> SamplingParams:
    """Build a vllm.SamplingParams from a model config block + runtime overrides.

    skip_special_tokens=False is hardcoded to preserve <think>, <|return|>,
    harmony channel tokens, etc. in `output.text` — required for downstream
    parsing of thinking-segment delimiters.
    """
    sp_cfg = model_config["sampling_params"]
    kwargs: dict[str, Any] = {
        "n": n_samples,
        "seed": seed,
        "skip_special_tokens": False,
    }
    for k in ("temperature", "top_p", "top_k", "max_tokens",
              "repetition_penalty", "presence_penalty", "frequency_penalty"):
        if k in sp_cfg:
            kwargs[k] = sp_cfg[k]
    if max_tokens_override is not None:
        kwargs["max_tokens"] = max_tokens_override
    return SamplingParams(**kwargs)
