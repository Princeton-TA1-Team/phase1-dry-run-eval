"""
kwdagger pipelines for the contextual-drag cards.

The card previously ran the whole six-step chain inside one node, which
meant any change re-ran everything: adjust the aggregate filter and the
expensive clean-round inference was recomputed even though its inputs were
untouched. As a DAG each stage caches on its own identity, so a filter
sweep reuses the generations.

    init_inference          <- clean prompt, the expensive step
          |
    eval_init               (--flatten_dataset)
          |
    postprocess
          |
    aggregate               <- may legitimately keep nothing
          |
    twof_inference          <- 2F prompt, the second expensive step
          |
    eval_twof               (--response_column twof_generations)
          |
    drag_summary            <- terminal artifact

``drag_summary`` takes two inputs, not one: the difference it reports is
between accuracies measured on the same surviving problems, so it needs
the aggregate's dataset as well as the 2F evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import kwdagger
from magnet.process_node import MagnetProcessNode

from cards.nodes.cd_aggregate import CDAggregateCLI
from cards.nodes.cd_drag_summary import CDDragSummaryCLI
from cards.nodes.cd_eval import CDEvalCLI
from cards.nodes.cd_inference import CDInferenceCLI
from cards.nodes.cd_postprocess import CDPostprocessCLI
from cards.nodes._load_result import load_node_result
from cards.nodes.cd_compose import CDComposeCLI
from cards.nodes.cd_ladder import CDLadderCLI
from cards.nodes.cd_route_loo import CDRouteLooCLI
from cards.nodes.cd_seeded_inference import CDSeededInferenceCLI

__all__ = ['drag_pipeline', 'route_model_pipeline']

# eval_models_params.json as it sits in the checkout. `resolve_lease_endpoints` runs
# in the *scheduler*, which is the one process in this pipeline that has only
# magnet + kwdagger -- every node's own dependency is satisfied inside its
# container. Importing ``contextual_drag`` to read this mapping therefore made
# DAG compilation depend on a package the scheduler has no reason to have, and
# the card crashed with ModuleNotFoundError on any host where it was not also
# pip-installed alongside magnet. It is a JSON file; read it as one.
_MODEL_PARAMS_RELPATH = Path(
    'src/contextual_drag/resources/inference/eval_models_params.json')


def _model_params_fpath(override=None) -> Path:
    """
    Locate eval_models_params.json without importing ``contextual_drag``.

    Args:
        override (str | None): explicit path, when a node names one.

    Returns:
        Path

    Raises:
        RuntimeError: if the packaged resource cannot be found. That is a
            broken checkout, not a card mistake, so it must be loud.
    """
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
        raise RuntimeError(
            f'model params file {candidate} does not exist')
    # cards/ sits directly under the repo root, next to src/.
    candidate = Path(__file__).resolve().parent.parent / _MODEL_PARAMS_RELPATH
    if candidate.exists():
        return candidate
    # Installed layout (no checkout): fall back to the packaged copy.
    try:
        from importlib import resources
        packaged = Path(str(resources.files(
            'contextual_drag.resources.inference'
        ).joinpath('eval_models_params.json')))
        if packaged.exists():
            return packaged
    except Exception:
        pass
    raise RuntimeError(
        f'cannot locate eval_models_params.json; looked for {candidate} and '
        'for the packaged contextual_drag.resources.inference copy. The card '
        'cannot resolve which endpoint its generation rounds need.')


class _Inference(MagnetProcessNode):
    """
    A generation round, holding its model only while it generates.

    The card names a *model config* (``Qwen3_8B_NoThinking``), not a served
    model, so the endpoint alias has to be looked up rather than read
    straight off a parameter. The alias is the config's ``served_model_name``
    when it has one, else ``model_name`` -- the same string the REST engine
    sends as ``model`` -- so an endpoint that serves this card is one the card
    can already address.
    """

    executable = 'python -m cards.nodes.cd_inference'
    params = CDInferenceCLI

    def load_result(self, node_dpath):
        # kwdagger has no generic loader for a Python ProcessNode;
        # without this, aggregate raises "'dict' object has no
        # attribute 'query_keys'" after every node has already run.
        return load_node_result(self, node_dpath)

    def resolve_lease_endpoints(self):
        config = self.final_config or {}
        alias = config.get('model_config')
        if not alias:
            return []
        fpath = _model_params_fpath(config.get('model_params_fpath'))
        blocks = json.loads(fpath.read_text())
        block = blocks.get(alias)
        if block is None:
            # An unknown alias is the node's problem to report when it runs,
            # with its own message naming the available aliases. Raising here
            # would turn it into an opaque DAG-compile failure. A *missing
            # file* is different and does raise, above -- that one is never
            # the card's fault.
            return []
        # served_model_name first: `model_name` is the HuggingFace repo id the
        # tokenizer is loaded from, and it doubles as the endpoint alias only
        # when the endpoint was named after its repo. Leasing the repo id from
        # a slug-named endpoint fails at acquire with "unknown endpoint".
        served = block.get('served_model_name') or block.get('model_name')
        return [served] if served else []


class _InitInference(_Inference):
    """Clean-prompt generation."""
    name = 'init_inference'


class _TwofInference(_Inference):
    """2F-augmented generation."""
    name = 'twof_inference'


class _EvalInit(MagnetProcessNode):
    """Score the clean round, emitting the flattened form."""
    name = 'eval_init'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI

    def load_result(self, node_dpath):
        # kwdagger has no generic loader for a Python ProcessNode;
        # without this, aggregate raises "'dict' object has no
        # attribute 'query_keys'" after every node has already run.
        return load_node_result(self, node_dpath)


class _EvalTwof(MagnetProcessNode):
    """Score the 2F round."""
    name = 'eval_twof'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI

    def load_result(self, node_dpath):
        # kwdagger has no generic loader for a Python ProcessNode;
        # without this, aggregate raises "'dict' object has no
        # attribute 'query_keys'" after every node has already run.
        return load_node_result(self, node_dpath)


class _Postprocess(MagnetProcessNode):
    """Fold flattened generations into a dataset."""
    name = 'postprocess'
    executable = 'python -m cards.nodes.cd_postprocess'
    params = CDPostprocessCLI

    def load_result(self, node_dpath):
        # kwdagger has no generic loader for a Python ProcessNode;
        # without this, aggregate raises "'dict' object has no
        # attribute 'query_keys'" after every node has already run.
        return load_node_result(self, node_dpath)


class _Aggregate(MagnetProcessNode):
    """Filter problems and build the 2F dataset."""
    name = 'aggregate'
    executable = 'python -m cards.nodes.cd_aggregate'
    params = CDAggregateCLI

    def load_result(self, node_dpath):
        # kwdagger has no generic loader for a Python ProcessNode;
        # without this, aggregate raises "'dict' object has no
        # attribute 'query_keys'" after every node has already run.
        return load_node_result(self, node_dpath)


class _DragSummary(MagnetProcessNode):
    """Compute the drag and emit the terminal artifact."""
    name = 'drag_summary'
    executable = 'python -m cards.nodes.cd_drag_summary'
    params = CDDragSummaryCLI

    def load_result(self, node_dpath):
        # kwdagger has no generic loader for a Python ProcessNode;
        # without this, aggregate raises "'dict' object has no
        # attribute 'query_keys'" after every node has already run.
        return load_node_result(self, node_dpath)


def drag_pipeline():
    """
    Build the contextual-drag DAG.

    Returns:
        kwdagger.Pipeline

    Example:
        >>> from cards.pipelines import drag_pipeline
        >>> sorted(drag_pipeline().nodes)
        ['aggregate', 'drag_summary', 'eval_init', 'eval_twof', 'init_inference', 'postprocess', 'twof_inference']
    """
    nodes = {
        'init_inference': _InitInference(),
        'eval_init': _EvalInit(),
        'postprocess': _Postprocess(),
        'aggregate': _Aggregate(),
        'twof_inference': _TwofInference(),
        'eval_twof': _EvalTwof(),
        'drag_summary': _DragSummary(),
    }

    nodes['init_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_init'].inputs['inference_manifest_fpath'])
    nodes['eval_init'].outputs['manifest_fpath'].connect(
        nodes['postprocess'].inputs['eval_manifest_fpath'])
    nodes['postprocess'].outputs['manifest_fpath'].connect(
        nodes['aggregate'].inputs['postprocess_manifest_fpath'])

    # The aggregate's manifest names the 2F dataset, so it is the second
    # round's input as well as the summary's.
    nodes['aggregate'].outputs['manifest_fpath'].connect(
        nodes['twof_inference'].inputs['data_fpath'])
    nodes['twof_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_twof'].inputs['inference_manifest_fpath'])

    nodes['aggregate'].outputs['manifest_fpath'].connect(
        nodes['drag_summary'].inputs['aggregate_manifest_fpath'])
    nodes['eval_twof'].outputs['manifest_fpath'].connect(
        nodes['drag_summary'].inputs['twof_eval_manifest_fpath'])

    # The two rounds share a model. Wire it rather than restating it, so a
    # model sweep is declared once.
    nodes['init_inference'].param_ports['model_config'].connect(
        nodes['twof_inference'].param_ports['model_config'])
    nodes['init_inference'].param_ports['model_config'].connect(
        nodes['aggregate'].param_ports['model_config'])

    dag = kwdagger.Pipeline(list(nodes.values()))
    dag.build_nx_graphs()
    return dag


# ---------------------------------------------------------------------------
# Route-model transfer card (Princeton-2, Phase I Final)
# ---------------------------------------------------------------------------
#
# One chain per model, fanned into a single leave-one-model-out node:
#
#     pool_inference      <- 16 clean answers per problem: the draft pool
#           |
#     eval_pool           (--flatten_dataset)
#           |
#     pool_postprocess    (reasoning trace stripped -> draft text)
#           |
#     compose             <- keeps problems with >= 4 right and >= 4 wrong;
#        /     \             builds T = 0..4 contexts x 8 orders
#       /       \
#  ladder_       baseline_inference   <- 16 FRESH answers per kept problem,
#  inference          |                  under a different seed
#      |         eval_baseline
#  eval_ladder        |
#       \       /
#        ladder           <- one ladder file per model
#           |
#           |  gather (group_by: []): every model's ladder, one manifest
#           v
#     route_loo           <- terminal artifact
#
# The three generation rounds share one model config, wired from
# pool_inference, so a model sweep is declared once and a model only ever
# sees drafts it wrote itself.


class _SeededInference(_Inference):
    executable = 'python -m cards.nodes.cd_seeded_inference'
    params = CDSeededInferenceCLI


class _PoolInference(_SeededInference):
    """Clean-slate pool: the drafts, and the problem selection."""
    name = 'pool_inference'


class _LadderInference(_SeededInference):
    """One response per (problem, rung, order) context."""
    name = 'ladder_inference'


class _BaselineInference(_SeededInference):
    """Fresh no-draft answers for the kept problems."""
    name = 'baseline_inference'


class _ResultNode(MagnetProcessNode):
    def load_result(self, node_dpath):
        # See _EvalInit: Python ProcessNodes need an explicit loader.
        return load_node_result(self, node_dpath)


class _EvalPool(_ResultNode):
    name = 'eval_pool'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _EvalLadder(_ResultNode):
    name = 'eval_ladder'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _EvalBaseline(_ResultNode):
    name = 'eval_baseline'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _PoolPostprocess(_ResultNode):
    name = 'pool_postprocess'
    executable = 'python -m cards.nodes.cd_postprocess'
    params = CDPostprocessCLI


class _Compose(_ResultNode):
    name = 'compose'
    executable = 'python -m cards.nodes.cd_compose'
    params = CDComposeCLI


class _Ladder(_ResultNode):
    name = 'ladder'
    executable = 'python -m cards.nodes.cd_ladder'
    params = CDLadderCLI


class _RouteLoo(_ResultNode):
    name = 'route_loo'
    executable = 'python -m cards.nodes.cd_route_loo'
    params = CDRouteLooCLI


def route_model_pipeline():
    """
    Build the route-model transfer DAG.

    Returns:
        kwdagger.Pipeline

    Example:
        >>> from cards.pipelines import route_model_pipeline
        >>> sorted(node.name for node in route_model_pipeline().nodes)
        ['baseline_inference', 'compose', 'eval_baseline', 'eval_ladder', 'eval_pool', 'ladder', 'ladder_inference', 'pool_inference', 'pool_postprocess', 'route_loo']
    """
    nodes = {
        'pool_inference': _PoolInference(),
        'eval_pool': _EvalPool(),
        'pool_postprocess': _PoolPostprocess(),
        'compose': _Compose(),
        'ladder_inference': _LadderInference(),
        'eval_ladder': _EvalLadder(),
        'baseline_inference': _BaselineInference(),
        'eval_baseline': _EvalBaseline(),
        'ladder': _Ladder(),
        'route_loo': _RouteLoo(),
    }

    nodes['pool_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_pool'].inputs['inference_manifest_fpath'])
    nodes['eval_pool'].outputs['manifest_fpath'].connect(
        nodes['pool_postprocess'].inputs['eval_manifest_fpath'])
    nodes['pool_postprocess'].outputs['manifest_fpath'].connect(
        nodes['compose'].inputs['postprocess_manifest_fpath'])

    nodes['compose'].outputs['manifest_fpath'].connect(
        nodes['ladder_inference'].inputs['data_fpath'])
    nodes['ladder_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_ladder'].inputs['inference_manifest_fpath'])

    nodes['compose'].outputs['baseline_manifest_fpath'].connect(
        nodes['baseline_inference'].inputs['data_fpath'])
    nodes['baseline_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_baseline'].inputs['inference_manifest_fpath'])

    nodes['compose'].outputs['manifest_fpath'].connect(
        nodes['ladder'].inputs['compose_manifest_fpath'])
    nodes['eval_ladder'].outputs['manifest_fpath'].connect(
        nodes['ladder'].inputs['ladder_eval_manifest_fpath'])
    nodes['eval_baseline'].outputs['manifest_fpath'].connect(
        nodes['ladder'].inputs['baseline_eval_manifest_fpath'])

    # Every model's ladder, as one newline-delimited manifest.
    nodes['ladder'].outputs['ladder_fpath'].connect(
        nodes['route_loo'].inputs['ladders_manifest_fpath'],
        gather={'group_by': [], 'order_by': ['model_config'],
                'require': 'all_success'})

    # One model per chain, declared once.
    for target in ('ladder_inference', 'baseline_inference', 'ladder'):
        nodes['pool_inference'].param_ports['model_config'].connect(
            nodes[target].param_ports['model_config'])

    dag = kwdagger.Pipeline(list(nodes.values()))
    dag.build_nx_graphs()
    return dag
