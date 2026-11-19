"""
Scheme 2: TrueSkill-based Game Simulation

Uses TrueSkill rating system instead of direct win/loss to build
the meta-payoff matrix. Handles uncertainty through sigma values.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from math import sqrt

# TrueSkill constants
MU_INITIAL = 25.0  # Initial mean skill
SIGMA_INITIAL = 25.0 / 3.0  # Initial uncertainty (std dev)
BETA = 25.0 / 6.0  # Performance variation
TAU = 25.0 / 300.0  # Dynamics factor (how much sigma grows)


@dataclass
class TrueSkillRating:
    """TrueSkill rating for a strategy."""
    mu: float = MU_INITIAL
    sigma: float = SIGMA_INITIAL
    name: str = ""


def v_function(draw_margin: float, delta: float, sigma: float) -> float:
    """V function for TrueSkill update (for games with draws)."""
    denom = sigma * sqrt(1 + (draw_margin ** 2) / (sigma ** 2))
    return np.exp(-delta * draw_margin) / denom


def w_function(draw_margin: float, delta: float, sigma: float) -> float:
    """W function for TrueSkill update (for games with draws)."""
    draw_margin_sq = draw_margin ** 2
    sigma_sq = sigma ** 2
    denom = 1 + draw_margin_sq / sigma_sq
    return (v_function(draw_margin, delta, sigma) ** 2 + np.exp(-draw_margin_sq / sigma_sq) / sigma_sq) / denom


def trueskill_update(
    winner_mu: float,
    winner_sigma: float,
    loser_mu: float,
    loser_sigma: float,
    draw_prob: float = 0.0,
    draw_margin: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Update TrueSkill ratings based on game outcome.

    Args:
        winner_mu, winner_sigma: Winner's current rating
        loser_mu, loser_sigma: Loser's current rating
        draw_prob: Probability of a draw (for future extension)
        draw_margin: Draw margin for games with draws

    Returns:
        Updated (winner_mu, winner_sigma, loser_mu, loser_sigma)
    """
    c = sqrt(winner_sigma ** 2 + loser_sigma ** 2 + 2 * BETA ** 2)

    # Delta = performance difference
    delta = winner_mu - loser_mu

    # PDF and CDF at delta/c
    from scipy.stats import norm
    pdf_val = norm.pdf(delta / c)
    cdf_val = norm.cdf(delta / c)

    # Update factors
    v = pdf_val / cdf_val if cdf_val > 1e-10 else 0.0
    w = v * (v + delta / c) / cdf_val if cdf_val > 1e-10 else 0.0

    # Update sigmas (apply dynamics factor)
    winner_sigma_new = sqrt(winner_sigma ** 2 + TAU ** 2) / sqrt(1 + winner_sigma ** 2 * w)
    loser_sigma_new = sqrt(loser_sigma ** 2 + TAU ** 2) / sqrt(1 + loser_sigma ** 2 * w)

    # Update mus
    winner_mu_new = winner_mu + winner_sigma_new ** 2 * v / c
    loser_mu_new = loser_mu - loser_sigma_new ** 2 * v / c

    return winner_mu_new, winner_sigma_new, loser_mu_new, loser_sigma_new


class Scheme2GS:
    """
    TrueSkill-based Game Simulation scheme.

    Uses TrueSkill rating system to maintain ratings for each strategy,
    with uncertainty (sigma) reflecting confidence in the rating.

    Maintains persistent ratings and only evaluates new strategy pairs.
    """

    def __init__(
        self,
        env_id: str = "TicTacToe-v0",
        num_games: int = 3
    ):
        self.env_id = env_id
        self.num_games = num_games
        self.textarena = TextArenaEnv(env_id=env_id, num_players=2)

        # Store ratings for each strategy
        self.ratings: Dict[str, TrueSkillRating] = {}

        # Track strategy order and prompts for incremental updates
        self._strategy_names: List[str] = []
        self._strategy_prompts: Dict[str, str] = {}

        # Payoff matrix cache
        self._payoff_matrix: Optional[np.ndarray] = None
        self._dirty = True  # Mark when ratings changed but matrix not updated

    @property
    def payoff_matrix(self) -> np.ndarray:
        """Get current payoff matrix based on ratings."""
        if self._payoff_matrix is None or self._dirty:
            self._rebuild_payoff_matrix()
        return self._payoff_matrix

    def _rebuild_payoff_matrix(self):
        """Rebuild payoff matrix from ratings."""
        n = len(self._strategy_names)
        self._payoff_matrix = np.zeros((n, n))
        for i, name_i in enumerate(self._strategy_names):
            for j, name_j in enumerate(self._strategy_names):
                if i == j:
                    self._payoff_matrix[i, j] = 0.0
                else:
                    mu_i = self.ratings[name_i].mu
                    mu_j = self.ratings[name_j].mu
                    self._payoff_matrix[i, j] = mu_i - mu_j
        self._dirty = False

    def get_or_create_rating(self, strategy_name: str) -> TrueSkillRating:
        """Get or create a TrueSkill rating for a strategy."""
        if strategy_name not in self.ratings:
            self.ratings[strategy_name] = TrueSkillRating(name=strategy_name)
        return self.ratings[strategy_name]

    def _make_agent(self, prompt: str) -> callable:
        """Create an agent function from a prompt."""
        def agent_fn(observation: str) -> str:
            return f"{prompt}\n\nCurrent observation: {observation}"
        return agent_fn

    def add_strategy(self, strategy_name: str, prompt: str) -> Dict[str, TrueSkillRating]:
        """
        Add a new strategy and evaluate it against all existing strategies.

        Args:
            strategy_name: Name of the new strategy
            prompt: Prompt of the new strategy

        Returns:
            Updated ratings dictionary
        """
        # Record the new strategy
        self._strategy_names.append(strategy_name)
        self._strategy_prompts[strategy_name] = prompt

        # Evaluate new strategy against all existing strategies
        for existing_name in self._strategy_names[:-1]:
            existing_prompt = self._strategy_prompts[existing_name]

            payoff_i, payoff_j = self.evaluate_pair(
                strategy_name,
                existing_name,
                prompt,
                existing_prompt,
                self._make_agent
            )

        # Mark payoff matrix as needing rebuild
        self._dirty = True

        return self.ratings

    def evaluate_pair(
        self,
        strategy_i: str,
        strategy_j: str,
        prompt_i: str,
        prompt_j: str,
        agent_fn: callable
    ) -> Tuple[float, float]:
        """
        Evaluate a pair of strategies and update TrueSkill ratings.

        Returns:
            (payoff_i, payoff_j) from game result
        """
        wins_i = 0
        wins_j = 0
        draws = 0

        for _ in range(self.num_games):
            env = TextArenaEnv(env_id=self.env_id, num_players=2)

            result = env.play_game(
                agents=[
                    self._make_agent(prompt_i),
                    self._make_agent(prompt_j)
                ]
            )

            if result.rewards[0] > result.rewards[1]:
                wins_i += 1
            elif result.rewards[0] < result.rewards[1]:
                wins_j += 1
            else:
                draws += 1

        # Update TrueSkill ratings
        rating_i = self.get_or_create_rating(strategy_i)
        rating_j = self.get_or_create_rating(strategy_j)

        if wins_i > wins_j:
            new_mu_i, new_sigma_i, new_mu_j, new_sigma_j = trueskill_update(
                rating_i.mu, rating_i.sigma,
                rating_j.mu, rating_j.sigma
            )
        elif wins_j > wins_i:
            new_mu_j, new_sigma_j, new_mu_i, new_sigma_i = trueskill_update(
                rating_j.mu, rating_j.sigma,
                rating_i.mu, rating_i.sigma
            )
        else:
            new_mu_i, new_sigma_i = rating_i.mu, rating_i.sigma
            new_mu_j, new_sigma_j = rating_j.mu, rating_j.sigma

        rating_i.mu, rating_i.sigma = new_mu_i, new_sigma_i
        rating_j.mu, rating_j.sigma = new_mu_j, new_sigma_j

        total = wins_i + wins_j + draws
        if total > 0:
            payoff_i = (wins_i - wins_j) / total
        else:
            payoff_i = 0.0

        return payoff_i, -payoff_i

    def build_payoff_matrix(
        self,
        strategies: List[Tuple[str, str]],  # List of (name, prompt)
        agent_fn: callable
    ) -> Tuple[np.ndarray, Dict[str, TrueSkillRating]]:
        """
        Build or extend payoff matrix and update TrueSkill ratings.

        If strategies have been added incrementally, only computes new entries.
        Existing entries are preserved.

        Returns:
            (payoff_matrix, ratings_dict)
        """
        # Check if we need to add new strategies
        for name, prompt in strategies:
            if name not in self._strategy_names:
                self.add_strategy(name, prompt)

        return self.payoff_matrix, self.ratings

    def get_meta_strategy(
        self,
        ratings: Optional[Dict[str, TrueSkillRating]] = None,
        method: str = "trueskill"
    ) -> np.ndarray:
        """
        Compute meta-strategy based on TrueSkill ratings.

        Args:
            ratings: Dictionary of TrueSkill ratings by strategy name.
                    If None, uses self.ratings.
            method: Method to use:
                - "trueskill": Softmax over mu values (skill-based)
                - "uniform": Uniform distribution
                - "last": Focus all weight on most recent strategy
                - "confidence": Weight by inverse sigma (confidence-based)

        Returns:
            Probability distribution over strategies
        """
        if ratings is None:
            ratings = self.ratings

        if not ratings:
            return np.array([])

        # Sort names to match strategy order
        if self._strategy_names:
            names = self._strategy_names
        else:
            names = list(ratings.keys())

        n = len(names)

        if method == "uniform":
            return np.ones(n) / n

        elif method == "last":
            meta = np.zeros(n)
            meta[-1] = 1.0
            return meta

        elif method == "confidence":
            sigmas = np.array([ratings[n].sigma for n in names])
            weights = 1.0 / (sigmas + 1e-6)
            return weights / weights.sum()

        elif method == "trueskill":
            mus = np.array([ratings[n].mu for n in names])
            temperature = 0.1
            exp_mu = np.exp((mus - max(mus)) / temperature)
            return exp_mu / exp_mu.sum()

        else:
            raise ValueError(f"Unknown meta-strategy method: {method}")

    def select_opponents(
        self,
        ratings: Optional[Dict[str, TrueSkillRating]] = None,
        strategy_name: str = None,
        top_k: int = 3
    ) -> List[str]:
        """
        Select most informative opponents for a given strategy.

        Selection based on:
        - High sigma (under-explored strategies)
        - Complementary strengths
        """
        if ratings is None:
            ratings = self.ratings

        if not ratings:
            return []

        if strategy_name is None:
            strategy_name = self._strategy_names[-1] if self._strategy_names else ""

        sorted_by_sigma = sorted(
            ratings.items(),
            key=lambda x: x[1].sigma,
            reverse=True
        )

        candidates = [n for n, _ in sorted_by_sigma if n != strategy_name]

        return candidates[:top_k]