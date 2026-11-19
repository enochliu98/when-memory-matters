"""
Scheme 1: Traditional Game Simulation (GS)

Uses TextArena for actual game simulation and direct win/loss recording
to build the meta-payoff matrix.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from ..envs.textarena_env import TextArenaEnv, GameResult


@dataclass
class Strategy:
    """Represents a strategy in the pool."""
    name: str
    prompt: str
    id: int


@dataclass
class MetaPayoffMatrix:
    """Meta-payoff matrix for strategy evaluation."""
    strategies: List[Strategy]
    payoff_matrix: np.ndarray  # [i, j] = payoff when strategy i plays against strategy j


def compute_nash_equilibrium(payoff_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Nash equilibrium for two-player zero-sum game.

    For zero-sum games, we find the minimax equilibrium:
    max_x min_y x^T * payoff * y

    Uses linear programming for the row player.

    Args:
        payoff_matrix: The payoff matrix

    Returns:
        Nash equilibrium mixed strategy as probability distribution
    """
    n = payoff_matrix.shape[0]

    # For zero-sum games, we solve:
    # max_x min_y x^T * M * y = min_y max_x x^T * M * y

    # Build the LP:
    # Variables: [p, v] where p is strategy distribution, v is the value
    # Maximize v subject to:
    #   M^T * p >= v (for all opponent strategies)
    #   sum(p) = 1
    #   p >= 0

    # Alternative: solve as linear assignment on negated matrix for opponent selection
    # Or use simpler approach with scipy.optimize

    # For small matrices, use iterative method
    # Fictitious play: both players best respond to opponent's historical distribution

    # Simple fictitious play
    nash_strategy = np.ones(n) / n
    opponent_strategy = np.ones(n) / n

    for _ in range(1000):
        # Row player's best response to opponent's strategy
        expected_payoffs = payoff_matrix @ opponent_strategy
        best_response = np.zeros(n)
        best_response[np.argmax(expected_payoffs)] = 1.0

        # Update opponent strategy
        opponent_strategy = 0.001 * best_response + 0.999 * opponent_strategy
        opponent_strategy /= opponent_strategy.sum()

        # Column player's best response to row player's strategy
        col_expected = payoff_matrix.T @ nash_strategy
        col_best_response = np.zeros(n)
        col_best_response[np.argmax(col_expected)] = 1.0

        # Update row player strategy
        nash_strategy = 0.001 * col_best_response + 0.999 * nash_strategy
        nash_strategy /= nash_strategy.sum()

    return nash_strategy


def compute_uniform_meta_strategy(payoff_matrix: np.ndarray) -> np.ndarray:
    """Compute uniform meta-strategy (equal probability for all strategies)."""
    n = payoff_matrix.shape[0]
    return np.ones(n) / n


def compute_last_strategy_meta(payoff_matrix: np.ndarray) -> np.ndarray:
    """
    Compute meta-strategy that focuses only on the most recent strategy.

    Returns probability 1.0 to the last (newest) strategy, 0.0 to others.
    """
    n = payoff_matrix.shape[0]
    meta = np.zeros(n)
    meta[-1] = 1.0
    return meta


class Scheme1GS:
    """
    Traditional Game Simulation scheme.

    For each pair of strategies, run actual games in TextArena
    and record win/loss/draw results.

    Maintains persistent payoff matrix and only computes new entries.
    """

    def __init__(
        self,
        env_id: str = "TicTacToe-v0",
        num_games: int = 5  # Number of games to average
    ):
        self.env_id = env_id
        self.num_games = num_games
        self.textarena = TextArenaEnv(env_id=env_id, num_players=2)

        # Persistent storage for incremental building
        self._strategies: List[Strategy] = []
        self._payoff_matrix: Optional[np.ndarray] = None

    @property
    def payoff_matrix(self) -> Optional[np.ndarray]:
        """Get current payoff matrix."""
        return self._payoff_matrix

    @property
    def strategies(self) -> List[Strategy]:
        """Get current strategies."""
        return self._strategies

    def _ensure_matrix_size(self, n: int):
        """Ensure payoff matrix is large enough for n strategies."""
        if self._payoff_matrix is None or self._payoff_matrix.shape[0] < n:
            if self._payoff_matrix is None:
                self._payoff_matrix = np.zeros((n, n))
            else:
                # Expand matrix
                old_matrix = self._payoff_matrix
                old_n = old_matrix.shape[0]
                new_matrix = np.zeros((n, n))
                new_matrix[:old_n, :old_n] = old_matrix
                self._payoff_matrix = new_matrix

    def evaluate_pair(
        self,
        strategy_i: Strategy,
        strategy_j: Strategy,
        agent_i_fn: callable,
        agent_j_fn: callable
    ) -> float:
        """
        Evaluate a pair of strategies against each other.

        Returns:
            Payoff for strategy_i (1 = win, 0 = draw, -1 = loss)
        """
        results = []

        for _ in range(self.num_games):
            env = TextArenaEnv(env_id=self.env_id, num_players=2)

            def make_agent(prompt: str):
                def agent_fn(observation: str) -> str:
                    return f"{prompt}\n\nCurrent observation: {observation}"
                return agent_fn

            result = env.play_game(
                agents=[
                    make_agent(strategy_i.prompt),
                    make_agent(strategy_j.prompt)
                ]
            )

            if result.rewards[0] > result.rewards[1]:
                results.append(1.0)
            elif result.rewards[0] < result.rewards[1]:
                results.append(-1.0)
            else:
                results.append(0.0)

        return np.mean(results)

    def add_strategy(self, strategy: Strategy) -> np.ndarray:
        """
        Add a new strategy and compute its interactions with existing strategies.

        Args:
            strategy: New strategy to add

        Returns:
            Row vector of payoffs for the new strategy against all existing strategies
        """
        def make_agent(prompt: str):
            def agent_fn(observation: str) -> str:
                return f"{prompt}\n\nCurrent observation: {observation}"
            return agent_fn

        n = len(self._strategies)
        self._ensure_matrix_size(n + 1)
        self._strategies.append(strategy)

        # Compute row for new strategy against all existing strategies
        for j in range(n):
            if j == n:
                self._payoff_matrix[n, j] = 0.0  # Same strategy = draw
            else:
                self._payoff_matrix[n, j] = self.evaluate_pair(
                    strategy,
                    self._strategies[j],
                    make_agent(strategy.prompt),
                    make_agent(self._strategies[j].prompt)
                )

        # Compute column for existing strategies against new strategy (symmetric)
        for i in range(n):
            # For zero-sum, payoff[i, n] = -payoff[n, i]
            self._payoff_matrix[i, n] = -self._payoff_matrix[n, i]

        return self._payoff_matrix[n, :]

    def build_payoff_matrix(
        self,
        strategies: List[Strategy],
        agent_fn: callable
    ) -> np.ndarray:
        """
        Build or extend the meta-payoff matrix.

        If strategies have been added incrementally, only computes new entries.
        Existing entries are preserved.

        Args:
            strategies: List of strategies to evaluate
            agent_fn: Function to create agent from strategy

        Returns:
            The full payoff matrix
        """
        n = len(strategies)

        # If matrix not initialized, build from scratch
        if self._payoff_matrix is None:
            self._payoff_matrix = np.zeros((n, n))
            self._strategies = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        self._payoff_matrix[i, j] = 0.0
                    else:
                        self._payoff_matrix[i, j] = self.evaluate_pair(
                            strategies[i],
                            strategies[j],
                            agent_fn(strategies[i]),
                            agent_fn(strategies[j])
                        )
            self._strategies = strategies.copy()
            return self._payoff_matrix

        # Check if we need to extend
        if n > len(self._strategies):
            # Add new strategies only
            for new_idx in range(len(self._strategies), n):
                new_strategy = strategies[new_idx]

                def make_agent(prompt: str):
                    def agent_fn(observation: str) -> str:
                        return f"{prompt}\n\nCurrent observation: {observation}"
                    return agent_fn

                self._ensure_matrix_size(new_idx + 1)
                self._strategies.append(new_strategy)

                # Row for new strategy
                for j in range(new_idx + 1):
                    if j == new_idx:
                        self._payoff_matrix[new_idx, j] = 0.0
                    else:
                        self._payoff_matrix[new_idx, j] = self.evaluate_pair(
                            new_strategy,
                            self._strategies[j],
                            make_agent(new_strategy.prompt),
                            make_agent(self._strategies[j].prompt)
                        )
                        # Symmetric entry
                        self._payoff_matrix[j, new_idx] = -self._payoff_matrix[new_idx, j]

        return self._payoff_matrix

    def get_meta_strategy(
        self,
        meta_payoff: np.ndarray,
        method: str = "nash"
    ) -> np.ndarray:
        """
        Compute meta-strategy from payoff matrix using specified method.

        Args:
            meta_payoff: The payoff matrix (or MetaPayoffMatrix object for compat)
            method: Method to use:
                - "uniform": Uniform distribution over all strategies
                - "last": Focus all weight on the most recent strategy
                - "nash": Nash equilibrium (zero-sum game solution)

        Returns:
            Probability distribution over strategies
        """
        if isinstance(meta_payoff, MetaPayoffMatrix):
            payoff = meta_payoff.payoff_matrix
        else:
            payoff = meta_payoff

        n = payoff.shape[0]

        if method == "uniform":
            return compute_uniform_meta_strategy(payoff)
        elif method == "last":
            return compute_last_strategy_meta(payoff)
        elif method == "nash":
            return compute_nash_equilibrium(payoff)
        else:
            raise ValueError(f"Unknown meta-strategy method: {method}")