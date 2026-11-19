"""
Best Response Solver (BRS)

Generates new strategies based on meta-strategy and opponent pool.
Follows the Reflexion approach: rollout -> analyze -> optimize.
"""

import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from ..prompts.base_prompts import BRS_PROMPT
from .scheme1 import Strategy


@dataclass
class BRSResult:
    """Result from BRS optimization."""
    original_strategy: Strategy
    problems: List[str]
    suggestions: List[str]
    improved_strategy: Strategy
    rollout_results: Optional[List[Dict]] = None
    rollout_trajectory: Optional[str] = None


class BestResponseSolver:
    """
    Best Response Solver for LangPSRO.

    Generates new strategies that respond to the current meta-strategy
    by:
    1. Rolling out against opponent strategies
    2. Analyzing failure cases
    3. Optimizing the strategy based on analysis
    """

    def __init__(self, llm_client: Optional[callable] = None):
        """
        Initialize BRS.

        Args:
            llm_client: LLM API client function.
                       Takes prompt string, returns response string.
        """
        self.llm_client = llm_client

    def generate_new_strategy(
        self,
        current_strategy: Strategy,
        opponent_pool: List[Strategy],
        meta_strategy: List[Tuple[str, float]],  # (name, probability)
        env_id: str = "TicTacToe-v0",
        num_rollout_games: int = 3
    ) -> BRSResult:
        """
        Generate an improved strategy.

        First runs rollouts against opponent strategies sampled from meta-strategy,
        then analyzes the results to optimize the strategy.

        Args:
            current_strategy: Strategy to improve
            opponent_pool: List of opponent strategies
            meta_strategy: Current meta-strategy distribution
            env_id: TextArena game ID for rollouts
            num_rollout_games: Number of games per opponent

        Returns:
            BRSResult with improved strategy
        """
        if self.llm_client is None:
            raise ValueError("LLM client is required for BRS")

        # Step 1: Run rollouts against opponent strategies
        rollout_results = self._run_rollouts(
            current_strategy=current_strategy,
            opponent_pool=opponent_pool,
            meta_strategy=meta_strategy,
            env_id=env_id,
            num_games=num_rollout_games
        )

        # Format rollout results as trajectory text
        trajectory_text = self._format_rollout_results(rollout_results)

        # Step 2: Format opponent pool (with meta-strategy weights)
        opponent_text = "\n".join([
            f"- {name} (weight: {weight:.2f}): {prompt[:200]}..."
            for name, weight in meta_strategy
            for _, prompt in [(next((s.name, s.prompt) for s in opponent_pool if s.name == name), ("", "")) if any(s.name == name for s in opponent_pool) else (name, "")]
        ])

        # Step 3: Format meta-strategy as text
        meta_text = "\n".join([
            f"- Strategy '{name}' with weight {weight:.2f}"
            for name, weight in meta_strategy
        ])

        # Step 4: Build prompt and call LLM
        prompt = BRS_PROMPT.format(
            current_strategy=current_strategy.prompt,
            opponent_pool=f"{opponent_text}\n\nMeta-strategy distribution:\n{meta_text}",
            rollout_trajectory=trajectory_text
        )

        # Strategy generation: allow slightly higher temperature for diversity
        # in the proposed improvements, but still constrained for reproducibility.
        response = self.llm_client(prompt, tag="brs_improve", temperature=0.5)

        # Parse response
        result = self._parse_response(response, current_strategy)
        result.rollout_results = rollout_results
        result.rollout_trajectory = trajectory_text

        return result

    def _run_rollouts(
        self,
        current_strategy: Strategy,
        opponent_pool: List[Strategy],
        meta_strategy: List[Tuple[str, float]],
        env_id: str,
        num_games: int = 3
    ) -> List[Dict]:
        """
        Run rollout games against opponent strategies.

        Samples opponents according to meta-strategy distribution.

        Args:
            current_strategy: Strategy to test
            opponent_pool: Available opponent strategies
            meta_strategy: Probability distribution over opponents
            env_id: TextArena game ID
            num_games: Number of games per opponent

        Returns:
            List of rollout result dicts
        """
        from ..envs.textarena_env import TextArenaEnv

        results = []

        # Create name-to-strategy map
        name_to_strategy = {s.name: s for s in opponent_pool}

        # Sample opponents according to meta-strategy
        opponent_names = [name for name, _ in meta_strategy]
        probabilities = [prob for _, prob in meta_strategy]

        for opponent_name, prob in zip(opponent_names, probabilities):
            if opponent_name not in name_to_strategy:
                continue

            opponent = name_to_strategy[opponent_name]

            for game_idx in range(num_games):
                env = TextArenaEnv(env_id=env_id, num_players=2)

                # Tag each trajectory step with role info so BRS analysis can
                # distinguish "the strategy under improvement took this move"
                # (is_brs_step=True, role="brs") from "the opponent did"
                # (is_brs_step=False, role="opponent"). The current strategy
                # sits in player slot 0 in the agents list below.
                result = env.play_game(
                    agents=[
                        self._make_agent(current_strategy.prompt, env_id=env_id),
                        self._make_agent(opponent.prompt, env_id=env_id)
                    ],
                    brs_player_id=0,
                    player_roles={0: "brs", 1: "opponent"},
                )

                # Determine outcome
                if result.rewards[0] > result.rewards[1]:
                    outcome = "WIN"
                elif result.rewards[0] < result.rewards[1]:
                    outcome = "LOSS"
                else:
                    outcome = "DRAW"

                # Use the per-step trajectory recorded by TextArenaEnv.play_game.
                # (Note: result.game_info does NOT contain a 'trajectory' key
                # in TextArena 0.7.x — it only has per-player metadata like
                # role/turn_count/reason. The real per-step trajectory lives
                # on GameResult.trajectory, recorded by play_game itself.)
                full_trajectory = result.trajectory if hasattr(result, 'trajectory') else []

                results.append({
                    "opponent": opponent_name,
                    "opponent_weight": prob,
                    "outcome": outcome,
                    "rewards": result.rewards,
                    "trajectory": full_trajectory,
                    "game_idx": game_idx,
                    "full_trajectory": full_trajectory,
                })

        return results

    def _format_rollout_results(self, rollout_results: List[Dict]) -> str:
        """Format rollout results as text."""
        if not rollout_results:
            return "No rollout results available."

        # Group by opponent for cleaner display
        opponent_stats = {}
        for result in rollout_results:
            opponent = result.get("opponent", "unknown")
            if opponent not in opponent_stats:
                opponent_stats[opponent] = {"wins": 0, "losses": 0, "draws": 0, "trajectories": []}

            outcome = result.get("outcome", "unknown")
            if outcome == "WIN":
                opponent_stats[opponent]["wins"] += 1
            elif outcome == "LOSS":
                opponent_stats[opponent]["losses"] += 1
            else:
                opponent_stats[opponent]["draws"] += 1

            traj = result.get("trajectory", [])
            if traj:
                # `result["trajectory"]` is a List[Dict] (one entry per step,
                # recorded by TextArenaEnv.play_game). Keep the full per-step
                # record so BRS can see exactly what the agent saw at each
                # decision point. All rollouts (num_rollout_games per opponent)
                # are forwarded to the prompt.
                opponent_stats[opponent]["trajectories"].append(traj)

        formatted = []
        for opponent, stats in opponent_stats.items():
            total = stats["wins"] + stats["losses"] + stats["draws"]
            win_rate = stats["wins"] / total if total > 0 else 0
            formatted.append(
                f"--- vs {opponent} ---\n"
                f"Record: {stats['wins']}W/{stats['losses']}L/{stats['draws']}D (win rate: {win_rate:.1%})\n"
                f"Trajectories:\n" + "\n".join(f"  - {t}" for t in stats["trajectories"])
            )

        return "\n\n".join(formatted)

    def _make_agent(
        self,
        prompt: str,
        env_id: str = "TicTacToe-v0",
        tag: str = "brs_rollout",
    ) -> callable:
        """
        Create an agent function from a strategy prompt.

        The agent formats its input into four parts before calling the LLM:
        1. Language Policy — the strategy prompt
        2. Local Observation — the current game state
        3. Instruction — guidance for action selection
        4. Output Format — game-specific action format (e.g. '[4]' for TicTacToe)

        Uses a low temperature (0.2) for stable action selection during rollout.
        The agent also parses the LLM's free-form output into a valid
        TextArena action string (e.g. extracts '[4]' from "Action: [4]").

        Args:
            prompt: Strategy prompt (language policy).
            env_id: TextArena env id.
            tag: Usage-tracking tag for the underlying LLM calls. Default
                "brs_rollout" — callers may override (e.g. "eval").
        """
        from ..envs.textarena_env import parse_action
        from ..utils.prompts import build_agent_prompt

        def agent_fn(observation: str) -> str:
            full_prompt = build_agent_prompt(
                strategy_prompt=prompt,
                observation=observation,
                env_id=env_id,
                include_thinking=False,
                memory_section=None,
                output_format_style="simple",
            )
            raw = self.llm_client(full_prompt, tag=tag, temperature=0.2)

            # print("-"*50)
            # print("【BR】")
            # print(raw)
            # print("-"*50)

            return parse_action(raw, env_id, observation)
        return agent_fn

    def _parse_response(
        self,
        response: str,
        original_strategy: Strategy
    ) -> BRSResult:
        """
        Parse the LLM's BRS response into a structured BRSResult.

        Expected response format (defined in BRS_PROMPT):

            [LESSONS LEARNED]
            - Failure mode 1: ...
            - Success pattern 1: ...

            [IMPROVED STRATEGY]
            <the improved strategy text>

        Parsing rules:
          * `[LESSONS LEARNED]` section: every bullet line (`- ...`) becomes
            an entry in `problems` if it starts with "Failure" / "failure" /
            "mistake" / "weakness", otherwise it goes into `suggestions`
            (which doubles as the "success patterns" bucket).
          * `[IMPROVED STRATEGY]` section: everything after the heading,
            trimmed, becomes the new strategy prompt. Falls back to the
            original strategy's prompt if the section is missing or empty.
        """
        problems: List[str] = []
        suggestions: List[str] = []
        improved_prompt = original_strategy.prompt

        # ---- [LESSONS LEARNED] section ----
        lessons_match = re.search(
            r'\[LESSONS\s+LEARNED\]\s*\n(.*?)(?=\[IMPROVED\s+STRATEGY\]|\Z)',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if lessons_match:
            for raw_line in lessons_match.group(1).splitlines():
                line = raw_line.strip()
                if not line.startswith("-"):
                    continue
                entry = line.lstrip("-").strip()
                if not entry:
                    continue
                lower = entry.lower()
                if lower.startswith(("failure", "mistake", "weakness", "problem", "issue")):
                    problems.append(entry)
                else:
                    # success patterns, observations, suggestions, etc.
                    suggestions.append(entry)

        # ---- [IMPROVED STRATEGY] section ----
        improved_match = re.search(
            r'\[IMPROVED\s+STRATEGY\]\s*\n(.*?)\Z',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if improved_match:
            body = improved_match.group(1).strip()
            # Strip leading/trailing code fences if the LLM added them
            body = re.sub(r"^```(?:prompt|text)?\s*\n?", "", body)
            body = re.sub(r"\n?```\s*$", "", body)
            body = body.strip()
            if body:
                improved_prompt = body

        # Create improved strategy with new ID
        improved_strategy = Strategy(
            name=f"{original_strategy.name}_v{original_strategy.id + 1}",
            prompt=improved_prompt,
            id=original_strategy.id + 1
        )

        return BRSResult(
            original_strategy=original_strategy,
            problems=problems,
            suggestions=suggestions,
            improved_strategy=improved_strategy
        )

    def run_rollout(
        self,
        strategy: Strategy,
        opponent: Strategy,
        env: 'TextArenaEnv',
        agent_fn: Optional[callable] = None
    ) -> Dict:
        """
        Run a single rollout game between strategy and opponent.

        DEPRECATED: Use _run_rollouts instead for full BRS workflow.

        Args:
            strategy: Strategy to test
            opponent: Opponent strategy
            env: TextArena environment
            agent_fn: Ignored (for backward compatibility)

        Returns:
            Dict with rollout results
        """
        result = env.play_game(
            agents=[
                self._make_agent(strategy.prompt),
                self._make_agent(opponent.prompt)
            ]
        )

        if result.rewards[0] > result.rewards[1]:
            outcome = "WIN"
        elif result.rewards[0] < result.rewards[1]:
            outcome = "LOSS"
        else:
            outcome = "DRAW"

        trajectory = ""
        if hasattr(result, 'game_info') and result.game_info:
            trajectory = str(result.game_info.get("trajectory", ""))

        return {
            "opponent": opponent.name,
            "outcome": outcome,
            "rewards": result.rewards,
            "trajectory": trajectory
        }