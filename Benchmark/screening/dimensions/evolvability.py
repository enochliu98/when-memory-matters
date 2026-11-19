"""
Evolvability Evaluator (权重 55%)

通过「专家策略 vs 初始策略」的直接对抗，测量游戏的可演化性。

核心思想：
  若一个人工设计的更强策略（Expert）能在对抗中显著压制初始策略（Naive），
  说明这个游戏存在可被利用的策略改进空间——即有可演化性。Expert 压得越狠，
  可演化性越高。

方法论：
  1. Expert vs Naive 打 N 局（双方各先手 N/2 局）
  2. 以「胜场差」(Expert 胜场 − Naive 胜场) 作为核心判据：
       margin_rate = margin / N,  范围 [−1, +1]
  3. margin_rate 强正 → Expert 明显强于 Naive → 可演化性高（评分高）
     margin_rate ≈ 0 → Expert 无法压过 Naive → 可演化性低（评分低）

评分标准（基于 margin_rate）：
  5 - margin_rate ≥ 0.50   Expert 碾压 Naive
  4 - 0.25 ≤ m < 0.50      Expert 明显强于 Naive
  3 - 0.10 ≤ m < 0.25      Expert 略强于 Naive
  2 - 0.03 ≤ m < 0.10      Expert 勉强领先
  1 - m < 0.03             Expert 无法压过 Naive（可演化性极低）
"""

from typing import Dict, Optional, Callable

_THINKING_MODE_CONFIG = {
    "thinking":         {"include_thinking": False, "output_format_style": "simple"},
    "non_thinking_pgt": {"include_thinking": True,  "output_format_style": "thinking_aware"},
    "non_thinking":     {"include_thinking": False, "output_format_style": "simple"},
}


_INITIAL_PROMPTS = {
    "TicTacToe-v0": (
        "You are a Tic Tac Toe player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
    "ConnectFour-v0": (
        "You are a Connect Four player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
    "KuhnPoker-v0": (
        "You are a Kuhn Poker player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
    "LiarsDice-v0": (
        "You are a Liar's Dice player. Your goal is to win the game. "
        "Play smart moves based on the current dice."
    ),
    "Othello-v0": (
        "You are an Othello player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
    "Breakthrough-v0": (
        "You are a Breakthrough player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
    "UltimateTicTacToe-v0": (
        "You are an Ultimate Tic Tac Toe player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
    "Nim-v0": (
        "You are a Nim player. Your goal is to win the game. "
        "Play smart moves based on the current board state."
    ),
}


_EXPERT_PROMPTS = {
    "TicTacToe-v0": (
        "You are a Tic Tac Toe player. Follow these rules in order:\n"
        "1. WIN: If you have two in a row, play the third to win.\n"
        "2. BLOCK: If opponent has two in a row, block them.\n"
        "3. FORK: Create a position where you have two winning threats.\n"
        "4. BLOCK FORK: If opponent can fork, force them to defend by creating your own threat.\n"
        "5. CENTER: Take the center square [4].\n"
        "6. OPPOSITE CORNER: If opponent is in a corner, take the opposite corner.\n"
        "7. EMPTY CORNER: Take any empty corner.\n"
        "8. EMPTY EDGE: Take any empty edge."
    ),
    "ConnectFour-v0": (
        "You are a Connect Four player. Follow these rules in order:\n"
        "1. WIN: If you have three in a row with an empty slot above, play there to win.\n"
        "2. BLOCK: If opponent has three in a row with an empty slot above, block them.\n"
        "3. SAFETY: Before any move, check that it doesn't give opponent a winning slot above your piece.\n"
        "4. CENTER: Prioritize the center columns (3, 4).\n"
        "5. STACK: Build vertical threats by stacking in the same column.\n"
        "6. TRAP: Create two threats simultaneously (fork)."
    ),
    "KuhnPoker-v0": (
        "You are a Kuhn Poker player. Follow these rules in order:\n"
        "1. CARDS: J < Q < K. K wins at showdown, J loses at showdown, Q splits.\n"
        "2. OPENING: With K → always bet. With Q → always bet (it dominates J). "
        "With J → always check (do not bluff at GTO).\n"
        "3. FACING A BET: With K → call. With Q → call. With J → fold.\n"
        "4. ROLES: Note which round it is. After 3 cards dealt without replacement, "
        "your hand information is more reliable — but the GTO play doesn't change.\n"
        "5. THINK: Count how many rounds are left. With J, the only profitable line "
        "is checking through; folding to a bet is the equilibrium call."
    ),
    "LiarsDice-v0": (
        "You are a Liar's Dice player. Follow these rules in order:\n"
        "1. COUNT: Tally your own dice by face value before deciding.\n"
        "2. OPENING: Open with [Bid: 1, F] where F is the face you have the most of.\n"
        "3. RAISE: When raising, prefer to bump quantity on the same face, or switch "
        "to a face you have more of. Avoid claiming face values you have NONE of in "
        "later rounds.\n"
        "4. CALL (Liar): If P(opponent's dice contain ≥ (bid_qty - my_match) of the "
        "face) is below ~30%, type [Call]. The expected value of a Call is positive "
        "when the bid is statistically unlikely.\n"
        "5. DICE COUNT: Track remaining dice for both players — fewer dice means "
        "tighter distributions and less room to bluff.\n"
        "6. THINK: Always consider the joint probability before raising or calling."
    ),
    "Othello-v0": (
        "You are an Othello player. Follow these strategic principles:\n"
        "1. CORNERS: Corners are the most valuable squares. Prioritize taking and threatening corners.\n"
        "2. MOBILITY: Minimize your opponent's legal moves while maximizing your own.\n"
        "3. EDGES: Stable edges (connected to corners) are valuable; avoid taking edges that give opponent corners.\n"
        "4. PARITY: In the endgame, control the parity of moves on each region.\n"
        "5. AVOID: Do not play to squares adjacent to empty corners (X-squares and C-squares) unless forced.\n"
        "6. EARLY: In the opening, prioritize moves that flip the fewest discs to maintain flexibility."
    ),
    "Breakthrough-v0": (
        "You are a Breakthrough player. Follow these principles:\n"
        "1. ADVANCE: Always advance your pawns forward. Pawns that don't advance are wasted.\n"
        "2. SUPPORT: Keep your pawns in a connected chain so they protect each other.\n"
        "3. BREAKTHROUGH: When you have a pawn that is ahead of the opponent's line, push it forward.\n"
        "4. WING: Attack on the wing where the opponent has fewer defenders.\n"
        "5. DEFENSE: Keep your back rank protected; don't leave gaps the opponent can exploit.\n"
        "6. TEMPO: Create multiple threats that force the opponent to react."
    ),
    "UltimateTicTacToe-v0": (
        "You are an Ultimate Tic Tac Toe player. Follow these principles:\n"
        "1. WIN MICRO: When you can win a micro-board, do it immediately.\n"
        "2. SENTE: Prefer moves that send the opponent to a micro-board where they have no advantage.\n"
        "3. CONTROL: Win the center micro-board [4] first — it controls the macro-center.\n"
        "4. FORCE: Send opponent to already-won or fully-drawn micro-boards to waste their turn.\n"
        "5. GLOBAL: Think about the macro-board, not just the current micro-board.\n"
        "6. AVOID: Don't give opponent easy wins on micro-boards they already control."
    ),
    "Nim-v0": (
        "You are a Nim player. Follow these principles strictly:\n"
        "1. NIM-SUM: Compute the XOR (nim-sum) of all pile sizes. If nim-sum != 0, you can win.\n"
        "2. WINNING: If nim-sum != 0, make a move that makes nim-sum = 0. Remove exactly the right number of objects from one pile.\n"
        "3. LOSING: If nim-sum = 0, any move leads to a winning position for the opponent. Make the move that gives the opponent the hardest-to-see position.\n"
        "4. SMALL PILES: When all piles have size 1, the winning move is to leave an odd number of piles.\n"
        "5. THINK: Calculate the nim-sum carefully before each move."
    ),
}


class EvolvabilityEvaluator:
    """
    Evaluates evolvability by pitting an expert strategy against a naive one.

    A stronger hand-designed prompt (Expert) plays the naive prompt (Naive)
    head-to-head. If Expert can clearly beat Naive, the game has exploitable
    strategy-improvement headroom — i.e. it is evolvable.
    """

    name = "evolvability"
    weight = 0.55

    def __init__(self, num_games: int = 10):
        self.num_games = num_games

    def evaluate(
        self,
        env_id: str,
        llm_client=None,
        game_wrapper=None,
        recorder=None,
        thinking_mode: str = "thinking",
        **kwargs,
    ) -> Dict:
        if llm_client is None or game_wrapper is None:
            return {
                "dimension": self.name,
                "weight": self.weight,
                "score": 0,
                "weighted_score": 0.0,
                "raw_metrics": {},
                "evidence": "SKIPPED: Requires LLM client + game wrapper.",
                "skipped": True,
            }

        initial_prompt = _INITIAL_PROMPTS.get(
            env_id,
            f"You are a {env_id} player. Your goal is to win the game."
        )
        expert_prompt = _EXPERT_PROMPTS.get(env_id)
        if expert_prompt is None:
            return {
                "dimension": self.name,
                "weight": self.weight,
                "score": 0,
                "weighted_score": 0.0,
                "raw_metrics": {},
                "evidence": f"SKIPPED: No expert prompt designed for {env_id}.",
                "skipped": True,
            }

        from langpsro.utils.prompts import build_agent_prompt
        from langpsro.envs.textarena_env import parse_action

        def make_agent(strategy_prompt):
            _thinking_cfg = _THINKING_MODE_CONFIG.get(thinking_mode, _THINKING_MODE_CONFIG["thinking"])
            def agent(obs):
                prompt_text = build_agent_prompt(
                    strategy_prompt=strategy_prompt,
                    observation=obs,
                    env_id=env_id,
                    include_thinking=_thinking_cfg["include_thinking"],
                    output_format_style=_thinking_cfg["output_format_style"],
                )
                raw = llm_client(prompt_text)
                return parse_action(raw, env_id, obs)
            return agent

        naive_agent = make_agent(initial_prompt)
        expert_agent = make_agent(expert_prompt)

        # Expert vs Naive — does a stronger prompt beat the naive one?
        # Expert is agent_p0, Naive is agent_p1; swap_first gives both sides
        # the first move, so this is a fair head-to-head.
        result = game_wrapper.play_multiple_games(
            expert_agent, naive_agent,
            num_games=self.num_games,
            swap_first=True,
            recorder=recorder,
            matchup_label="evolv_expert",
            agent_labels=("expert", "naive"),
        )
        expert_wins = result["p0_wins"]
        naive_wins = result["p1_wins"]
        draws = result["draws"]
        num_games = result["total_games"]

        # 胜场差 (win margin): Expert − Naive. Normalised to [−1, +1].
        margin = expert_wins - naive_wins
        margin_rate = margin / num_games if num_games > 0 else 0.0
        score = self._compute_score(margin_rate)

        raw_metrics = {
            "expert_win_count": expert_wins,
            "naive_win_count": naive_wins,
            "draw_count": draws,
            "num_games": num_games,
            "expert_win_rate": result["p0_win_rate"],
            "naive_win_rate": result["p1_win_rate"],
            "draw_rate": result["draw_rate"],
            "margin": margin,
            "margin_rate": round(margin_rate, 3),
            "opponent": "llm_with_naive_prompt",
            "initial_prompt": initial_prompt,
            "expert_prompt": expert_prompt,
        }

        evidence = (
            f"Expert vs Naive: {expert_wins}胜 {naive_wins}负 {draws}平, "
            f"胜场差 {margin:+d} (margin_rate {margin_rate:+.0%}). "
            f"评分: {score}/5."
        )

        return {
            "dimension": self.name,
            "weight": self.weight,
            "score": score,
            "weighted_score": score * self.weight,
            "raw_metrics": raw_metrics,
            "evidence": evidence,
            "skipped": False,
        }

    def _compute_score(self, margin_rate: float) -> int:
        """Score evolvability from the Expert-vs-Naive win margin.

        margin_rate = (Expert_wins - Naive_wins) / N, range [−1, +1].
        Strongly positive ⇒ Expert clearly beats Naive ⇒ high evolvability.
        """
        if margin_rate >= 0.50:
            return 5
        if margin_rate >= 0.25:
            return 4
        if margin_rate >= 0.10:
            return 3
        if margin_rate >= 0.03:
            return 2
        return 1
