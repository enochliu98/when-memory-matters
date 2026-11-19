"""
Main Runner for LangPSRO

Example usage of the LangPSRO framework.
"""

from typing import List, Callable, Optional
import numpy as np

from .schemes.scheme1 import Scheme1GS, Strategy, MetaPayoffMatrix
from .schemes.scheme2 import Scheme2GS
from .schemes.scheme4 import Scheme4Clustering
from .schemes.brs import BestResponseSolver
from .prompts.base_prompts import BASE_PROMPT


class LangPSRO:
    """
    Main LangPSRO runner.

    Implements the full PSRO loop with LLM-based strategies.
    """

    def __init__(
        self,
        llm_client: Callable[[str], str],
        scheme: int = 1,
        env_id: str = "TicTacToe-v0",
        max_strategies: int = 10,
        meta_strategy_method: str = "nash"
    ):
        """
        Initialize LangPSRO.

        Args:
            llm_client: LLM API client function
            scheme: Which scheme to use (1, 2, or 4)
            env_id: TextArena game ID
            max_strategies: Maximum number of strategies to generate
            meta_strategy_method: Method for computing meta-strategy:
                - "uniform": Uniform distribution
                - "last": Focus on most recent strategy
                - "nash": Nash equilibrium (scheme 1 only)
                - "trueskill": Softmax over mu (scheme 2 only)
                - "confidence": Inverse sigma weighting (scheme 2 only)
        """
        self.llm_client = llm_client
        self.scheme = scheme
        self.env_id = env_id
        self.max_strategies = max_strategies
        self.meta_strategy_method = meta_strategy_method

        # Initialize strategy pool
        self.strategy_pool: List[Strategy] = []

        # Initialize components based on scheme
        if scheme in [1, 2]:
            self.game_sim = Scheme1GS(env_id=env_id) if scheme == 1 else Scheme2GS(env_id=env_id)
        elif scheme == 4:
            self.clustering = Scheme4Clustering(llm_client=llm_client)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        self.brs = BestResponseSolver(llm_client=llm_client)

    def initialize(self, base_prompt: str = BASE_PROMPT):
        """Initialize with base strategy."""
        strategy = Strategy(
            name="init_strategy",
            prompt=base_prompt,
            id=0
        )
        self.strategy_pool.append(strategy)

    def run_iteration(self, meta_strategy: Optional[np.ndarray] = None) -> Strategy:
        """
        Run one iteration of PSRO.

        Args:
            meta_strategy: Current meta-strategy distribution

        Returns:
            Newly generated strategy
        """
        if len(self.strategy_pool) == 0:
            raise RuntimeError("Strategy pool is empty. Call initialize() first.")

        # Build meta-strategy if not provided
        if meta_strategy is None:
            meta_strategy = self._compute_meta_strategy()

        # Build opponent pool
        opponent_pool = self.strategy_pool.copy()

        # Get current strategy (last one added)
        current = self.strategy_pool[-1]

        # Run BRS to generate new strategy
        meta_strategy_list = [
            (s.name, float(p))
            for s, p in zip(self.strategy_pool, meta_strategy)
        ]

        result = self.brs.generate_new_strategy(
            current_strategy=current,
            opponent_pool=opponent_pool,
            meta_strategy=meta_strategy_list,
            env_id=self.env_id
        )

        # Add new strategy to pool
        new_strategy = result.improved_strategy
        new_strategy.id = len(self.strategy_pool)
        self.strategy_pool.append(new_strategy)

        return new_strategy

    def _compute_meta_strategy(self) -> np.ndarray:
        """Compute meta-strategy based on current scheme and configured method."""
        n = len(self.strategy_pool)

        if self.scheme == 1:
            meta_payoff = self.game_sim.build_payoff_matrix(
                self.strategy_pool,
                self._make_agent
            )
            return self.game_sim.get_meta_strategy(meta_payoff, method=self.meta_strategy_method)

        elif self.scheme == 2:
            strategy_tuples = [(s.name, s.prompt) for s in self.strategy_pool]
            payoff, ratings = self.game_sim.build_payoff_matrix(
                strategy_tuples,
                self._make_agent
            )
            return self.game_sim.get_meta_strategy(ratings, method=self.meta_strategy_method)

        elif self.scheme == 4:
            strategy_tuples = [(s.name, s.prompt) for s in self.strategy_pool]
            cluster_result = self.clustering.cluster_strategies(strategy_tuples)
            return self.clustering.get_meta_strategy(cluster_result)

        return np.ones(n) / n

    def _make_agent(self, strategy: Strategy) -> Callable:
        """Create agent function from strategy."""
        def agent(observation: str) -> str:
            return f"{strategy.prompt}\n\nCurrent observation: {observation}"
        return agent

    def run(
        self,
        num_iterations: int = 5,
        base_prompt: str = BASE_PROMPT
    ) -> List[Strategy]:
        """
        Run the full PSRO loop.

        Args:
            num_iterations: Number of iterations to run
            base_prompt: Initial strategy prompt

        Returns:
            Final strategy pool
        """
        # Initialize
        self.initialize(base_prompt)

        # Run iterations
        for i in range(num_iterations):
            if len(self.strategy_pool) >= self.max_strategies:
                break

            meta_strategy = self._compute_meta_strategy()

            new_strategy = self.run_iteration(meta_strategy)

            print(f"Iteration {i+1}: Added strategy '{new_strategy.name}'")

        return self.strategy_pool


def example_llm_client(prompt: str) -> str:
    """
    Example LLM client for testing.

    Replace this with actual API call.
    """
    # Placeholder - replace with actual API call
    return "This is a placeholder response. Connect your LLM API here."


if __name__ == "__main__":
    # Example usage
    langpsro = LangPSRO(
        llm_client=example_llm_client,
        scheme=1,
        env_id="TicTacToe-v0",
        max_strategies=5
    )

    final_pool = langpsro.run(num_iterations=3)

    print(f"\nFinal strategy pool ({len(final_pool)} strategies):")
    for s in final_pool:
        print(f"  - {s.name}: {s.prompt[:100]}...")