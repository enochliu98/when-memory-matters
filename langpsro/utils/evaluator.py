"""
Strategy Evaluator for LangPSRO

Evaluates strategy performance against baseline and records results.
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from ..envs.textarena_env import TextArenaEnv, GameResult
from ..schemes.scheme1 import Strategy


@dataclass
class EvaluationResult:
    """Result from evaluating a strategy."""
    strategy_name: str
    opponent_name: str
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    avg_reward: float
    game_trajectories: List[Dict] = None  # Raw game trajectories

    def __post_init__(self):
        if self.game_trajectories is None:
            self.game_trajectories = []


@dataclass
class RoundResult:
    """Result from a round of PSRO evaluation."""
    round_num: int
    strategies: List[Dict]
    meta_strategy: List[float]
    evaluation_results: List[EvaluationResult]
    timestamp: str


class StrategyEvaluator:
    """
    Evaluates strategies using a baseline agent.

    For fair comparison, the baseline uses a simple fixed prompt.
    """

    def __init__(
        self,
        baseline_prompt: str,
        llm_client,
        env_id: str = "TicTacToe-v0",
        num_games: int = 10
    ):
        """
        Initialize evaluator.

        Args:
            baseline_prompt: Simple prompt for baseline agent
            llm_client: LLM client for strategy-based agents
            env_id: TextArena game ID
            num_games: Number of games to play per evaluation
        """
        self.baseline_prompt = baseline_prompt
        self.llm_client = llm_client
        self.env_id = env_id
        self.num_games = num_games

    def create_agent(
        self,
        strategy_prompt: str,
        env_id: Optional[str] = None,
        tag: str = "eval",
    ):
        """
        Create an agent function from a strategy prompt.

        The agent formats its input into four parts before calling the LLM:
        1. Language Policy — the strategy prompt
        2. Local Observation — the current game state
        3. Instruction — guidance for action selection
        4. Output Format — game-specific action format (e.g. '[4]' for TicTacToe)

        After the LLM responds, the agent parses the free-form text into a
        valid TextArena action string (e.g. extracts '[4]' from "Action: [4]").

        Args:
            strategy_prompt: The strategy prompt (language policy).
            env_id: TextArena env id (defaults to self.env_id).
            tag: Usage-tracking tag for the underlying LLM calls. Default
                "eval" — callers may override.
        """
        from ..envs.textarena_env import parse_action
        from .prompts import build_agent_prompt

        env_id = env_id or self.env_id

        def agent(observation: str) -> str:
            prompt = build_agent_prompt(
                strategy_prompt=strategy_prompt,
                observation=observation,
                env_id=env_id,
                include_thinking=False,
                memory_section=None,
                output_format_style="simple",
            )
            raw = self.llm_client(prompt, tag=tag)

            # print("-"*50)
            # print("【BR】")
            # print(raw)
            # print("-"*50)

            return parse_action(raw, env_id, observation)
        return agent

    def evaluate_strategy(
        self,
        strategy: Strategy,
        opponent_prompt: str,
        num_games: Optional[int] = None,
        record_trajectories: bool = True
    ) -> EvaluationResult:
        """
        Evaluate one strategy against an opponent.

        Args:
            strategy: Strategy to evaluate
            opponent_prompt: Opponent's strategy prompt
            num_games: Number of games (uses default if None)
            record_trajectories: Whether to record game trajectories

        Returns:
            EvaluationResult
        """
        num_games = num_games or self.num_games
        wins, losses, draws = 0, 0, 0
        total_reward = 0.0
        game_trajectories = []

        for _ in range(num_games):
            env = TextArenaEnv(env_id=self.env_id, num_players=2)

            strategy_agent = self.create_agent(strategy.prompt)
            opponent_agent = self.create_agent(opponent_prompt)

            result = env.play_game(agents=[strategy_agent, opponent_agent])

            if result.rewards[0] > result.rewards[1]:
                wins += 1
            elif result.rewards[0] < result.rewards[1]:
                losses += 1
            else:
                draws += 1
            total_reward += result.rewards[0]

            if record_trajectories and result.trajectory:
                game_trajectories.append({
                    "trajectory": result.trajectory,
                    "outcome": "WIN" if result.rewards[0] > result.rewards[1] else "LOSS" if result.rewards[0] < result.rewards[1] else "DRAW",
                    "rewards": result.rewards
                })

        return EvaluationResult(
            strategy_name=strategy.name,
            opponent_name="baseline",
            games_played=num_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / num_games,
            loss_rate=losses / num_games,
            draw_rate=draws / num_games,
            avg_reward=total_reward / num_games,
            game_trajectories=game_trajectories
        )

    def evaluate_pool(
        self,
        strategies: List[Strategy],
        opponent_prompt: Optional[str] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate all strategies in a pool against the baseline.

        Args:
            strategies: List of strategies to evaluate
            opponent_prompt: Opponent prompt (uses baseline if None)

        Returns:
            List of evaluation results
        """
        opponent = opponent_prompt or self.baseline_prompt
        results = []

        for strategy in strategies:
            result = self.evaluate_strategy(strategy, opponent)
            results.append(result)

        return results


class ExperimentLogger:
    """
    Logs experiment results to a directory.

    Each run gets its own subdirectory with timestamp.
    """

    def __init__(
        self,
        base_dir: str = "experiments",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize experiment logger.

        Args:
            base_dir: Base directory for all experiments
            experiment_name: Optional name for this experiment
        """
        self.base_dir = base_dir
        self.experiment_name = experiment_name or "langpsro"
        self.run_dir: Optional[str] = None
        self.round_results: List[RoundResult] = []

    def create_run_dir(self) -> str:
        """
        Create a new run directory with timestamp.

        Returns:
            Path to the new run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.experiment_name}_{timestamp}"
        self.run_dir = os.path.join(self.base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def log_round(self, round_result: RoundResult):
        """
        Log results from a round.

        Args:
            round_result: Result from the round
        """
        self.round_results.append(round_result)

        # Save individual round file
        round_file = os.path.join(
            self.run_dir,
            f"round_{round_result.round_num:03d}.json"
        )
        with open(round_file, 'w') as f:
            json.dump(asdict(round_result), f, indent=2)

    def save_summary(self, token_usage: Optional[Dict] = None):
        """
        Save a summary of all rounds.

        Args:
            token_usage: Optional dict describing LLM token usage for the
                whole run (e.g. the output of `client.get_usage_summary()`).
                When provided, it is written to `token_usage.json` next to
                `summary.json`.
        """
        summary = {
            "experiment_name": self.experiment_name,
            "total_rounds": len(self.round_results),
            "rounds": [asdict(r) for r in self.round_results]
        }

        summary_file = os.path.join(self.run_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        if token_usage is not None:
            usage_file = os.path.join(self.run_dir, "token_usage.json")
            with open(usage_file, 'w') as f:
                json.dump(token_usage, f, indent=2)

    def get_run_dir(self) -> str:
        """Get the current run directory."""
        if self.run_dir is None:
            self.create_run_dir()
        return self.run_dir


# Per-env baseline prompts. These are intentionally simple, generic
# strategies used as the "fixed opponent" in evaluation. Each baseline
# only knows the legal move format and the most basic objective — no
# game-specific heuristics, so it serves as a fair, low-skill opponent
# across all games.
BASELINE_PROMPTS: Dict[str, str] = {
    "TicTacToe-v0": (
        "You are a Tic Tac Toe player.\n\n"
        "Your goal is to win the game or achieve a draw.\n\n"
        "Simple rules:\n"
        "- Place your mark in an available cell (0-8)\n"
        "- Try to get three in a row\n"
        "- Block your opponent's winning moves\n"
    ),
    "ConnectFour-v0": (
        "You are a Connect Four player.\n\n"
        "Your goal is to connect four of your discs vertically, horizontally, "
        "or diagonally before your opponent does.\n\n"
        "Simple rules:\n"
        "- Drop your disc into an available column (0-6)\n"
        "- Watch the column heights and avoid playing into a full column\n"
        "- Try to build your own lines of three while blocking opponent "
        "lines of three\n"
    ),
    "2048-v0": (
        "You are a 2048 player.\n\n"
        "Your goal is to combine tiles to reach higher values; the game ends "
        "when no moves are possible.\n\n"
        "Simple rules:\n"
        "- Reply with one move: 0=up, 1=down, 2=left, 3=right\n"
        "- Prefer moves that keep your largest tile in a corner\n"
        "- Avoid moves that would leave the board with no legal shifts\n"
    ),
}


def get_baseline_prompt(env_id: str) -> str:
    """
    Return the baseline prompt for a given TextArena env id.

    Falls back to a generic "follow the game rules" prompt for unknown
    envs so unknown games can still run an evaluation.
    """
    if env_id in BASELINE_PROMPTS:
        return BASELINE_PROMPTS[env_id]
    return (
        f"You are a player of the TextArena game '{env_id}'.\n\n"
        "Your goal is to win the game.\n\n"
        "Simple rules:\n"
        "- Always follow the format specified in the observation\n"
        "- Only make legal moves\n"
        "- Try to win; if you cannot, minimize your losses\n"
    )


# Backward-compat constant. Existing imports keep working (TicTacToe is
# the default), but new code should call get_baseline_prompt(env_id).
BASELINE_PROMPT = BASELINE_PROMPTS["TicTacToe-v0"]