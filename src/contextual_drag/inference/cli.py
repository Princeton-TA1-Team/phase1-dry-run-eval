"""ModalCLI surface for `contextual_drag inference {run, list_models}`.

Per-cell only — mirrors PR-2's vllm_cli.py but dispatches into our async
vLLM driver (AsyncLLMEngine + prompt-hash JSONL resume).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import scriptconfig as scfg


def _resolve_enable_thinking(enable: bool, disable: bool) -> bool | None:
    """PR-2 tri-state: --enable_thinking / --disable_thinking / unset → True/False/None."""
    if enable and disable:
        raise SystemExit("--enable_thinking and --disable_thinking are mutually exclusive")
    if enable:
        return True
    if disable:
        return False
    return None


def _cfg_to_namespace(cfg) -> SimpleNamespace:
    """scriptconfig DataConfig → SimpleNamespace the driver expects."""
    data = dict(cfg)
    ns = SimpleNamespace(**data)
    ns.enable_thinking = _resolve_enable_thinking(
        getattr(cfg, "enable_thinking", False),
        getattr(cfg, "disable_thinking", False),
    )
    return ns


class InferenceRunCLI(scfg.DataConfig):
    """Run the async vLLM driver on one inference cell."""
    __command__ = "run"

    # PR-2 per-cell surface
    model_config = scfg.Value("Qwen3_8B_NoThinking",
                               help="Model configuration alias from eval_models_params.json.",
                               tags=["algo_param"])
    config_path = scfg.Value(None,
                              help="Override eval_models_params.json path (default: packaged resource).",
                              tags=["algo_param"])
    data_path = scfg.Value(None, required=True,
                            help="Path to the HF dataset directory (.ds) to process.",
                            tags=["algo_param"])
    output_dir = scfg.Value("./outputs",
                             help="Directory for completions.jsonl (created if absent).",
                             tags=["algo_param"])
    prompt_template_path = scfg.Value(None, required=True,
                                       help="Path to the prompt template JSON.",
                                       tags=["algo_param"])
    prompt_template_key = scfg.Value(None, required=True,
                                      help="Template key inside the template JSON.",
                                      tags=["algo_param"])
    task_name = scfg.Value("init_response",
                            help="Output column prefix: `<task_name>_generations`, "
                                 "`<task_name>_prompt`, etc.",
                            tags=["algo_param"])
    n = scfg.Value(1, type=int, help="Number of responses per prompt.",
                   tags=["algo_param"])
    max_tokens = scfg.Value(None, type=int,
                             help="Override SamplingParams.max_tokens from the model config.",
                             tags=["algo_param"])
    max_questions = scfg.Value(None, type=int,
                                help="Cap on rows to process after resume filtering.",
                                tags=["algo_param"])
    batch_size = scfg.Value(32, type=int,
                             help="(Surface-compat) Driver dispatches by max_concurrent; "
                                  "this value is recorded but does not gate the engine.",
                             tags=["algo_param"])
    tensor_parallel_size = scfg.Value(1, type=int,
                                       help="Number of GPUs for tensor parallelism.",
                                       tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.95, type=float,
                                         help="GPU memory utilization ratio.",
                                         tags=["algo_param"])
    seed = scfg.Value(42, type=int, help="Sampling seed.", tags=["algo_param"])
    enable_thinking = scfg.Value(False, isflag=True, help="Enable thinking mode.",
                                  tags=["algo_param"])
    disable_thinking = scfg.Value(False, isflag=True, help="Disable thinking mode.",
                                   tags=["algo_param"])
    # Async-engine reliability knobs
    max_concurrent = scfg.Value(128, type=int,
                                 help="Cap on in-flight requests fed into the engine.",
                                 tags=["algo_param"])
    prefix_caching = scfg.Value(True, isflag=True, help="Toggle vLLM prefix caching.",
                                 tags=["algo_param"])
    per_sample_requests = scfg.Value(True, isflag=True,
                                      help="Issue n separate n=1 requests per row "
                                           "(avoids the gpt-oss-family sub-sample drop bug).",
                                      tags=["algo_param"])
    gdn_prefill_backend = scfg.Value(None, choices=["flashinfer", "triton"],
                                      help="Override GDN prefill backend.",
                                      tags=["algo_param"])
    shutdown_grace = scfg.Value(60, type=float,
                                 help="Seconds to wait for in-flight requests on SIGTERM.",
                                 tags=["algo_param"])
    status_interval = scfg.Value(30, type=float,
                                  help="Seconds between periodic [status] lines.",
                                  tags=["algo_param"])
    dry_run = scfg.Value(False, isflag=True,
                          help="Resolve + prescan + preview prompts; exit before engine init.",
                          tags=["algo_param"])

    @classmethod
    def main(cls, argv=True, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.inference.run_model import amain

        ns = _cfg_to_namespace(cfg)
        # `n` and `max_questions` map onto the driver's internal names.
        ns.n_samples = ns.n
        ns.max_rows_per_cell = ns.max_questions
        asyncio.run(amain(ns))


class InferenceDryRunCLI(InferenceRunCLI):
    """Dry-run: resolve work item + preview prompt; never load the engine."""
    __command__ = "dry_run"

    dry_run = scfg.Value(True, isflag=True,
                          help="Forced True for the dry_run verb.",
                          tags=["algo_param"])


class InferenceListModelsCLI(scfg.DataConfig):
    """Print every model alias defined in the packaged eval_models_params.json."""
    __command__ = "list_models"

    @classmethod
    def main(cls, argv=True, **kwargs):
        cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.config.resources import inference_model_config_resource

        for name in sorted(inference_model_config_resource()):
            print(name)


class InferenceCLI(scfg.ModalCLI):
    run = InferenceRunCLI
    dry_run = InferenceDryRunCLI
    list_models = InferenceListModelsCLI
