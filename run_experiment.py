"""
Run LangPSRO experiment with evaluation and logging.

Usage:
    python run_experiment.py --scheme 1 --num_iterations 5 --num_games 5
"""

import argparse
import os
from datetime import datetime

from langpsro import LangPSRO, BASE_PROMPT
from langpsro.utils import (
    make_llm_client,
    StrategyEvaluator,
    ExperimentLogger,
    get_baseline_prompt,
)


def run_experiment(
    scheme: int = 1,
    num_iterations: int = 5,
    num_games: int = 5,
    env_id: str = "TicTacToe-v0",
    max_strategies: int = 10,
    meta_strategy_method: str = "nash",
    llm_provider: str = "volcengine",
):
    """
    Run a LangPSRO experiment with evaluation.

    Args:
        scheme: Which scheme to use (1, 2, or 4)
        num_iterations: Number of PSRO iterations
        num_games: Number of games per evaluation
        env_id: TextArena game ID
        max_strategies: Maximum strategies in pool
        meta_strategy_method: Method for computing meta-strategy:
            - "uniform": Uniform distribution over all strategies
            - "last": Focus all weight on most recent strategy
            - "nash": Nash equilibrium (scheme 1 only)
            - "trueskill": Softmax over mu (scheme 2 only)
            - "confidence": Inverse sigma weighting (scheme 2 only)
    """
    print(f"=" * 60)
    print(f"LangPSRO Experiment")
    print(f"=" * 60)
    print(f"Scheme: {scheme}")
    print(f"Meta-strategy method: {meta_strategy_method}")
    print(f"Iterations: {num_iterations}")
    print(f"Evaluation games per matchup: {num_games}")
    print(f"Game: {env_id}")
    print(f"=" * 60)

    # Initialize LLM client
    # Default temperature 0.2 is used for rollout/evaluation (stable actions);
    # BRS strategy generation overrides per-call with temperature=0.5
    # (see langpsro/schemes/brs.py). The provider is selectable via
    # `llm_provider` ("volcengine" by default; pass "deepseek" to use
    # the native DeepSeek endpoint instead).
    print(f"Using LLM provider: {llm_provider}")
    client = make_llm_client(llm_provider, temperature=0.2)

    # Pick the right baseline prompt for this env (TicTacToe / ConnectFour /
    # 2048 / etc). The baseline is the fixed low-skill opponent used in
    # evaluation across all rounds.
    baseline_prompt = get_baseline_prompt(env_id)
    print(f"Using baseline prompt for {env_id}:\n{baseline_prompt}\n")

    # Initialize evaluator and logger
    evaluator = StrategyEvaluator(
        baseline_prompt=baseline_prompt,
        llm_client=client,
        env_id=env_id,
        num_games=num_games
    )

    logger = ExperimentLogger(
        base_dir="experiments",
        experiment_name=f"langpsro_scheme{scheme}_{meta_strategy_method}"
    )
    logger.create_run_dir()
    print(f"\nExperiment directory: {logger.get_run_dir()}")

    # Initialize LangPSRO
    langpsro = LangPSRO(
        llm_client=client,
        scheme=scheme,
        env_id=env_id,
        max_strategies=max_strategies,
        meta_strategy_method=meta_strategy_method
    )

    # Initialize with base prompt
    langpsro.initialize(base_prompt=BASE_PROMPT)

    # Local imports kept at module top would be cleaner, but using
    # here to keep them co-located with the only consumer.
    from langpsro.utils.evaluator import RoundResult
    from dataclasses import asdict

    def _make_round_result(
        round_num: int,
        strategies_pool,
        meta_strategy,
        eval_results,
    ) -> RoundResult:
        strategies_data = [
            {"name": s.name, "prompt": s.prompt, "id": s.id}
            for s in strategies_pool
        ]
        return RoundResult(
            round_num=round_num,
            strategies=strategies_data,
            meta_strategy=(meta_strategy.tolist() if meta_strategy is not None else []),
            evaluation_results=[asdict(r) for r in eval_results],
            timestamp=datetime.now().isoformat(),
        )

    # Evaluate initial strategy
    print(f"\n--- Round 0: Initial Strategy ---")
    eval_results = evaluator.evaluate_pool(langpsro.strategy_pool, baseline_prompt)
    print(f"Initial strategy evaluation:")
    for result in eval_results:
        print(f"  vs baseline: {result.wins}W/{result.losses}L/{result.draws}D "
              f"(win rate: {result.win_rate:.1%})")

    # Log round 0 (no meta-strategy yet — there is only one strategy).
    logger.log_round(_make_round_result(
        round_num=0,
        strategies_pool=langpsro.strategy_pool,
        meta_strategy=None,
        eval_results=eval_results,
    ))

    # Run iterations
    for i in range(num_iterations):
        print(f"\n--- Round {i+1}: PSRO Iteration ---")

        # Compute meta-strategy and run BRS
        meta_strategy = langpsro._compute_meta_strategy()
        new_strategy = langpsro.run_iteration(meta_strategy)

        print(f"Generated new strategy: {new_strategy.name}")

        # Evaluate only the latest strategy against baseline
        eval_results = evaluator.evaluate_pool([new_strategy], baseline_prompt)

        print(f"\nLatest strategy evaluation (vs baseline):")
        for result in eval_results:
            print(f"  {result.strategy_name}: {result.wins}W/{result.losses}L/{result.draws}D "
                  f"(win rate: {result.win_rate:.1%})")

        # Log round results
        round_result = _make_round_result(
            round_num=i + 1,
            strategies_pool=langpsro.strategy_pool,
            meta_strategy=meta_strategy,
            eval_results=eval_results,
        )
        logger.log_round(round_result)

    # Save summary
    token_usage_summary = client.get_usage_summary()
    logger.save_summary(token_usage=token_usage_summary)

    # Print token usage breakdown
    print(f"\n" + "=" * 60)
    print(f"LLM Token Usage")
    print(f"=" * 60)
    total = token_usage_summary["total"]
    print(f"  Total: {total['call_count']} calls, "
          f"{total['prompt_tokens']} prompt + {total['completion_tokens']} completion "
          f"= {total['total_tokens']} tokens"
          + (f" (reasoning: {total['reasoning_tokens']})" if total.get("reasoning_tokens") else ""))
    if token_usage_summary["by_tag"]:
        print(f"  By tag:")
        for tag, stats in token_usage_summary["by_tag"].items():
            print(f"    {tag:>16s}: {stats['call_count']:>4d} calls, "
                  f"{stats['prompt_tokens']:>7d} prompt + {stats['completion_tokens']:>6d} completion "
                  f"= {stats['total_tokens']:>7d} tokens"
                  + (f" (reasoning: {stats['reasoning_tokens']})" if stats.get("reasoning_tokens") else ""))
    print(f"=" * 60)

    print(f"\n" + "=" * 60)
    print(f"Experiment complete!")
    print(f"Results saved to: {logger.get_run_dir()}")
    print(f"=" * 60)

    return langpsro.strategy_pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangPSRO experiment")
    parser.add_argument("--scheme", type=int, default=1, choices=[1, 2, 4],
                        help="Scheme to use (1=traditional GS, 2=TrueSkill, 4=LLM clustering)")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of PSRO iterations")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games per evaluation")
    parser.add_argument("--env_id", type=str, default="TicTacToe-v0",
                        help="TextArena game ID")
    parser.add_argument("--max_strategies", type=int, default=10,
                        help="Maximum strategies in pool")
    parser.add_argument("--meta_strategy_method", type=str, default="uniform",
                        choices=["uniform", "last", "nash", "trueskill", "confidence"],
                        help="Method for computing meta-strategy")
    parser.add_argument("--llm", type=str, default="volcengine",
                        choices=["volcengine", "deepseek"],
                        help="LLM provider to use (default: volcengine)")

    args = parser.parse_args()

    run_experiment(
        scheme=args.scheme,
        num_iterations=args.num_iterations,
        num_games=args.num_games,
        env_id=args.env_id,
        max_strategies=args.max_strategies,
        meta_strategy_method=args.meta_strategy_method,
        llm_provider=args.llm,
    )
