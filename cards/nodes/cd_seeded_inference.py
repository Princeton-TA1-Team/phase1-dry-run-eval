"""
An inference round with an explicit sampling seed.

``inference run`` seeds sample ``i`` of every row with ``seed + i + 1``, so
two rounds over the same prompts with the same seed return the same samples.
The route-model card samples the same problems twice -- once to build the
draft pool and select problems, once to estimate the baseline -- and the plan
requires the baseline to be *new* answers: reusing the pool's would carry the
selection into the estimate and bias the baseline toward the selection
window. The two rounds therefore run with different seeds, declared here.

A subclass, not a param on ``CDInferenceCLI``, so the existing cards keep
their node identities and their caches.
"""
from __future__ import annotations

import scriptconfig as scfg

from cards.nodes.cd_inference import CDInferenceCLI


class CDSeededInferenceCLI(CDInferenceCLI):
    """Run one inference cell with a declared sampling seed."""

    seed = scfg.Value(
        42, type=int,
        help=('Sampling seed passed to `inference run --seed`. 42 is that '
              "command's own default, so an unset seed reproduces it."),
        tags=['algo_param'])

    @classmethod
    def _extra_run_args(cls, config) -> list:
        _check_endpoint_window(config)
        return ['--seed', int(config.seed)]


def _check_endpoint_window(config):
    """
    Refuse an endpoint whose context window is smaller than the model config's.

    The route-model card's generation budget is the model config's whole
    context window, as in the runs its source ladders came from. A served
    endpoint with a smaller window either rejects long requests or ends
    generations earlier than those runs did -- a different measurement, not a
    noisier one. vLLM reports its window as ``max_model_len`` in
    ``GET /v1/models``; when it is smaller, this stops the node with a message
    that says so. An endpoint that does not report a window is trusted, and
    no endpoint at all means the in-process engine, which is built with the
    config's window.
    """
    import json
    import os
    import urllib.request
    from pathlib import Path

    base = (os.environ.get('CONTEXTUAL_DRAG_ENDPOINT', '').strip()
            or os.environ.get('OPENAI_BASE_URL', '').strip())
    if not base:
        return
    from cards.pipelines import _model_params_fpath
    blocks = json.loads(Path(_model_params_fpath(
        getattr(config, 'model_params_fpath', None))).read_text())
    block = blocks.get(config.model_config) or {}
    want = int(block.get('context_length') or 0)
    served = (os.environ.get('CONTEXTUAL_DRAG_ENDPOINT_MODEL', '').strip()
              or block.get('served_model_name') or block.get('model_name'))
    if not want or not served:
        return
    request = urllib.request.Request(base.rstrip('/') + '/models')
    key = (os.environ.get('CONTEXTUAL_DRAG_ENDPOINT_API_KEY')
           or os.environ.get('OPENAI_API_KEY'))
    if key:
        request.add_header('Authorization', f'Bearer {key}')
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            models = json.loads(response.read()).get('data') or []
    except Exception as ex:  # an unreachable listing is not a window
        print(f'[route-model] could not list endpoint models ({ex!r}); '
              f'not checking its context window', flush=True)
        return
    for entry in models:
        if entry.get('id') != served:
            continue
        have = entry.get('max_model_len')
        if have is not None and int(have) < want:
            raise SystemExit(
                f'endpoint serves {served} with max_model_len={have}, but '
                f'{config.model_config} declares context_length={want}. This '
                f'card generates up to the full window, as the runs behind '
                f'its source ladders did; serve the model with '
                f'max_model_len >= {want}.')
        return


__cli__ = CDSeededInferenceCLI

if __name__ == '__main__':
    CDSeededInferenceCLI.main()
