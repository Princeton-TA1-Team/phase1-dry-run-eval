"""Magnet node: recursive self-improvement (naive) — N independent trajectories, averaged.

Thin wrapper over `cards.nodes._recursive_multirun.run_node`. Builds the round-0
init pool once, launches `n_runs` independent recursive runs (run_id 1..N, each
sampling `n_samples_solve` rollouts/step and picking one to continue), and reports
per-step pass@1 averaged over (n_runs x n_samples_solve) generations with a std
across the N trajectories. Faithfully matches the old-repo deploy_sweep procedure.

result.json keys: acc_round_0, acc_round_max (means across runs), delta_acc_naive (PASS = deterioration, claim direction <=),
per_step_pass_at_1 {step: {mean,std,n_runs}}, n_runs, n_runs_completed,
aggregate_failed, makeup_exhausted, max_step_reached, n_problems_round_0, ...
Sentinel -1.0 replaces null (magnet Symbol.eval cannot resolve null).
"""
from __future__ import annotations

import scriptconfig as scfg


class RunRecursiveNaiveCLI(scfg.DataConfig):
    """Sweep-able pipeline node for the recursive (naive) self-improvement card."""

    model_config = scfg.Value("GPT_OSS_20B_recursive", tags=["algo_param"])
    init_alias = scfg.Value("GPT_OSS_20B", tags=["algo_param"])
    task = scfg.Value("aime24", tags=["algo_param"],
                      choices=["aime24", "aime25", "hmmt24", "hmmt25"])
    data_path = scfg.Value("data/full_data/aime24/aime24.ds", tags=["algo_param"])
    template_path = scfg.Value(
        "prompt_templates/1f_templates.json", tags=["algo_param"])
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json", tags=["algo_param"])
    init_n_samples = scfg.Value(8, type=int, tags=["algo_param"])
    max_recursive_steps = scfg.Value(16, type=int, tags=["algo_param"])
    n_samples_solve = scfg.Value(16, type=int, tags=["algo_param"],
                                 help="Rollouts sampled per step per run (pick one to continue).")
    n_runs = scfg.Value(16, type=int, tags=["algo_param"],
                        help="Independent trajectories (run_id 1..N); per-step accuracy "
                             "is averaged over n_runs x n_samples_solve generations.")
    seed = scfg.Value(42, type=int, tags=["algo_param"],
                      help="Rollout-generation seed (fixed across runs, per old-repo).")
    tensor_parallel_size = scfg.Value(4, type=int, tags=["algo_param"],
                                      help="GPUs per run. Runs go num_gpus//tp at a time.")
    gpu_memory_utilization = scfg.Value(0.9, type=float, tags=["algo_param"])
    max_tokens = scfg.Value(65536, type=int, tags=["algo_param"])

    results_fpath = scfg.Value("results.json", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        from cards.nodes._recursive_multirun import run_node
        run_node(cfg, variant="naive", delta_key="delta_acc_naive")


def main(argv=None):
    return RunRecursiveNaiveCLI.main(argv=argv)


if __name__ == "__main__":
    main()
