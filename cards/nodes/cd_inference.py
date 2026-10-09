"""
One inference round: render prompts and generate responses.

Used twice in the drag pipeline -- once on the clean prompt and once on
the 2F-augmented prompt -- as two nodes sharing this executable.

``model_params_fpath`` is an ALGO param, and adding it MOVED EVERY EXISTING
CARD'S NODE-ID HASH -- ``init_inference`` on the scale-up card went from
``is88uqzijfj4`` to ``a8zhr41w051o``. Generations cached under the old ids are
not reused, so the first run of any pre-existing card after this change
regenerates. That is a real cost, paid once, and it is the correct one: the
file the alias is looked up in determines which weights answer, so two runs
that disagree about it are not the same measurement and must not share a cache
entry. Demoting it to a perf param would keep the hashes still by lying about
what it does.

``gpu_memory_utilization`` and ``tensor_parallel_size`` are perf params,
not algo params. They change how the work is placed on hardware, never
what is produced, so they must not enter the node's identity: raising the
memory fraction should not invalidate generations. That was wrong in the
monolithic card, where ``gpu_memory_utilization`` was declared an algo
param and so was part of the card's hash. They are also ignored entirely
when generation goes through a REST endpoint.
"""

from __future__ import annotations

# `shutil` is used on the resume path, to discard a partial attempt before
# retrying. It fires only when output_dir exists without a manifest, which is
# why a first run never reaches it and the missing import went unnoticed.
import shutil
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import (
    first_match, read_manifest, run_contextual_drag, write_manifest,
)


class CDInferenceCLI(scfg.DataConfig):
    """Run one inference cell."""

    model_config = scfg.Value(
        'Qwen3_8B_NoThinking',
        help='Alias from eval_models_params.json.',
        tags=['algo_param'])

    model_params_fpath = scfg.Value(
        None, help=(
            'Override the eval_models_params.json the alias is looked up in. '
            'None (the default) uses the packaged resource, so existing cards '
            'are unaffected. This completes a seam the pipeline already had: '
            '`_model_params_fpath` in cards/pipelines.py has always read this '
            'key off the node config to resolve the endpoint, but the node '
            'never declared it, so the override could not be set from a card '
            'and the scheduler and the node could not be pointed at the same '
            'file. Forwarded to `inference run --config_path`, so the alias '
            'resolves identically on both sides.'),
        tags=['algo_param'])

    data_fpath = scfg.Value(
        None,
        help=('Dataset to run on. Either a .ds path (the clean round) or an '
              'upstream manifest naming one (the 2F round).'),
        tags=['in_path'])

    template_path = scfg.Value(
        'prompt_templates/init_response_prompt_templates.json',
        help='JSON file of prompt templates.', tags=['algo_param'])
    template_key = scfg.Value(
        'qa_mc_prompt', help='Template key within that file.',
        tags=['algo_param'])
    task_name = scfg.Value(
        'init_response', help='Names the generated columns.',
        tags=['algo_param'])

    thinking = scfg.Value(
        None, help=(
            'Whether the chat template renders a thinking preamble: True, '
            'False, or None to leave it to contextual_drag (which defaults '
            'to on). Applied at prompt-render time, not as a sampling param.'),
        tags=['algo_param'])

    max_questions = scfg.Value(
        8, type=int, help='Cap on dataset rows (0 = whole dataset).',
        tags=['algo_param'])
    n = scfg.Value(8, type=int, help='Samples per question.',
                   tags=['algo_param'])
    max_tokens = scfg.Value(2048, type=int, help='Per-response budget.',
                            tags=['algo_param'])

    # Placement, not behaviour -- see the module docstring.
    gpu_memory_utilization = scfg.Value(
        0.85, type=float, help='vLLM GPU memory fraction. Ignored on a REST '
                               'endpoint.', tags=['perf_param'])
    tensor_parallel_size = scfg.Value(
        1, type=int, help='vLLM tensor parallelism. Ignored on a REST '
                          'endpoint.', tags=['perf_param'])

    manifest_fpath = scfg.Value(
        'inference.json', help='Manifest naming the completions produced.',
        tags=['out_path', 'primary'])

    @classmethod
    def _extra_run_args(cls, config) -> list:
        """
        Further ``inference run`` flags a subclass declares params for.

        A hook rather than a new param here: declaring a param on this class
        moves the node-id hash of every card that uses it (see the module
        docstring), so a flag only some cards need lives on a subclass.
        """
        return []

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        manifest_fpath = Path(config.manifest_fpath).resolve()
        output_dir = manifest_fpath.parent / 'inference'
        # The manifest is this node's done-marker: the scheduler skips a node
        # whose manifest exists, so if we are running, the previous attempt did
        # not finish. Its partial output must go. `inference run` APPENDS to
        # completions.jsonl, so a killed run left 4 rows that the retry added
        # 128 more to -- 132 rows for 128 problems, 2 of them duplicated.
        # Grading then died on ragged arrays, three hours later and nowhere
        # near the cause:
        #
        #   ValueError: setting an array element with a sequence ... (132,)
        if output_dir.exists() and not manifest_fpath.exists():
            stale = [f for f in output_dir.rglob('*') if f.is_file()]
            if stale:
                print(f'[{config.task_name}] discarding {len(stale)} file(s) '
                      f'from an unfinished earlier attempt in {output_dir}')
                shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_fpath, n_rows = _resolve_dataset(config.data_fpath)
        if dataset_fpath is None:
            # Upstream produced nothing usable. Record it and stop; the
            # summary node turns this into a legible card outcome.
            write_manifest(manifest_fpath, dataset_fpath=None, n_rows=0,
                           completions_fpath=None, output_dir=output_dir,
                           task_name=config.task_name, skipped=True,
                           reason='upstream produced no usable dataset')
            return

        max_questions = int(config.max_questions)
        if n_rows is not None and n_rows > 0:
            max_questions = min(max_questions, n_rows) if max_questions else n_rows
        elif not max_questions:
            # `0` means "all" only when n_rows is known, and _resolve_dataset
            # returns None for a raw dataset directory -- so on a first round
            # 0 means zero. Generating nothing then wrote a manifest reporting
            # success, because the every-request-failed check below is gated on
            # max_questions > 0. Silence is the wrong answer either way: say
            # the count is needed rather than run an empty round.
            raise SystemExit(
                f'{config.task_name}: max_questions=0 means "all rows", but '
                f'the row count of {config.data_fpath} is unknown (a dataset '
                f'directory does not report one; only an upstream manifest '
                f'does). Give max_questions an explicit number for this round.'
            )

        # contextual_drag reads --enable_thinking / --disable_thinking as a
        # tri-state: neither flag means "unset", which its prescan turns into
        # thinking ON. Passing neither is therefore not neutral, and this node
        # passed neither -- so an alias named `Qwen3_8B_NoThinking` produced
        # exactly the same prompts as `Qwen3_8B_Thinking`. Nothing downstream
        # of the alias name ever acted on it.
        thinking = _coerce_tristate(config.thinking)
        thinking_args = []
        if thinking is not None:
            thinking_args = ['--enable_thinking' if thinking
                             else '--disable_thinking']

        # Only passed when set: `inference run` defaults config_path to None,
        # meaning the packaged resource, and a '--config_path' with an empty
        # value would be a path of '' rather than a default.
        #
        # _coerce_optional_path, not a truth test, and for the same reason
        # _coerce_tristate exists below. kwdagger renders EVERY declared param
        # into the job script, so a card that never mentions this one still
        # gets `--model_params_fpath=None`, which makes a round trip through
        # the shell. scriptconfig's smartcast turns that back into None today
        # -- verified -- but the string 'None' is truthy, so if that parse rule
        # ever changes, a plain truth test sends `--config_path None` and every
        # card that never asked for an override dies opening a file called
        # None. Normalizing costs nothing and does not depend on the rule.
        params_fpath = _coerce_optional_path(config.model_params_fpath)
        params_args = []
        if params_fpath is not None:
            params_args = ['--config_path', params_fpath]

        run_contextual_drag([
            'inference', 'run',
            '--model_config', config.model_config,
            '--data_path', dataset_fpath,
            '--prompt_template_path', config.template_path,
            '--prompt_template_key', config.template_key,
            '--output_dir', output_dir,
            '--task_name', config.task_name,
            '--max_questions', max_questions,
            '--n', config.n,
            '--batch_size', min(8, max(1, max_questions)),
            '--tensor_parallel_size', config.tensor_parallel_size,
            '--gpu_memory_utilization', config.gpu_memory_utilization,
            '--max_tokens', config.max_tokens,
        ] + params_args + thinking_args + cls._extra_run_args(config))

        completions = first_match(output_dir, 'completions.jsonl')

        # `contextual_drag inference run` exits 0 even when every row failed --
        # it reports per-row errors and moves on, which is right for a partial
        # run and wrong for a total one. Without this check a node that
        # generated NOTHING wrote a healthy-looking manifest, and the failure
        # surfaced two nodes downstream as `AxisError(-1, 0, None)` from inside
        # numpy, naming neither the endpoint nor the request that was rejected.
        #
        # Seen for real: the 2F round's prompts totalled one token over the
        # endpoint's max_model_len, so all six rows came back 400
        # ContextWindowExceededError and the run was declared complete.
        if max_questions > 0 and _n_completions(completions) == 0:
            raise SystemExit(
                f'{config.task_name}: generated 0 completions for '
                f'{max_questions} row(s). Every request failed -- the errors '
                f'above name the cause. Common ones: the prompt plus '
                f'max_tokens exceeds the endpoint\'s max_model_len, or the '
                f'endpoint is serving a different model than the card names.')

        write_manifest(manifest_fpath, dataset_fpath=dataset_fpath,
                       n_rows=max_questions, completions_fpath=completions,
                       output_dir=output_dir, task_name=config.task_name,
                       skipped=False)


def _n_completions(fpath):
    """
    Count records in a completions JSONL, tolerating absence.

    Args:
        fpath (str | Path | None): the file, or None when none was produced.

    Returns:
        int

    Example:
        >>> _n_completions(None)
        0
    """
    if not fpath:
        return 0
    path = Path(fpath)
    if not path.exists():
        return 0
    with open(path) as file:
        return sum(1 for line in file if line.strip())


def _coerce_optional_path(value):
    """
    Normalize an optional path that may arrive as the string ``'None'``.

    kwdagger renders every declared parameter into the generated job script, so
    a card that does not set this one still produces
    ``--model_params_fpath=None``. scriptconfig currently smartcasts that back
    to ``None``, but that is a parse rule rather than a guarantee, and the
    string ``'None'`` is truthy -- under any rule that preserved it, a plain
    truth test would send ``--config_path None`` on every card that never asked
    for an override. Also normalizes the empty string, which no rule converts.

    Args:
        value: a path, None, or a string spelling of either.

    Returns:
        str | None

    Example:
        >>> [_coerce_optional_path(v) for v in [None, 'None', '', 'none', '/a/b']]
        [None, None, None, None, '/a/b']
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {'', 'none', 'null', 'unset'}:
        return None
    return text


def _coerce_tristate(value):
    """
    Normalize a True/False/unset parameter that may arrive as a string.

    kwdagger renders matrix entries into a job script as ``--thinking=False``,
    so the value makes a round trip through the shell. scriptconfig's smartcast
    turns that back into a bool today, but it is one parse rule away from
    arriving as the string ``'False'`` -- which is truthy, and would silently
    enable thinking on every card that asked for it to be off. Since that is
    the precise failure this parameter exists to fix, normalize rather than
    trust a truth test.

    Args:
        value: bool, None, or a string spelling of either.

    Returns:
        bool | None

    Example:
        >>> [_coerce_tristate(v) for v in [True, 'False', 'none', None, '']]
        [True, False, None, None, None]
    """
    if isinstance(value, bool) or value is None:
        return value
    text = str(value).strip().lower()
    if text in {'', 'none', 'null', 'auto', 'unset'}:
        return None
    if text in {'1', 'true', 'yes', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(
        f'thinking={value!r} is not True, False, or unset')


def _resolve_dataset(data_fpath):
    """
    Resolve the input, which is either a dataset or an upstream manifest.

    Returns:
        tuple: ``(dataset_fpath | None, n_rows | None)``.
    """
    path = Path(data_fpath)
    if path.is_dir():
        return path, None
    manifest = read_manifest(path)
    dataset = manifest.get('dataset_fpath')
    return (Path(dataset) if dataset else None), manifest.get('n_kept')


__cli__ = CDInferenceCLI

if __name__ == '__main__':
    CDInferenceCLI.main()
