"""
Non-Saturation Evaluator (权重 35%)

通过真实 LLM 对局测量非饱和性。

核心逻辑：
  1. LLM vs Optimal 打 N 局（双方各先手 N/2 局）
  2. 以「胜场差」(LLM 胜场 − Optimal 胜场) 作为核心判据：
       margin_rate = margin / N,  范围 [−1, +1]
     - margin_rate 强负 → LLM 远低于最优 → 头部空间大 → 非饱和性高（评分高）
     - margin_rate ≈ 0（或几乎全平局）→ LLM 已接近最优 → 饱和（评分低）
  3. 顺带收集真实 trajectory 数据 (步数 + token 消耗) 供 CostEvaluator 使用

评分标准（基于 margin_rate；draw_rate 仅在极端平局时判饱和）：
  5 - margin_rate ≤ −0.50   LLM 被最优碾压
  4 - −0.50 < m ≤ −0.25     LLM 明显低于最优
  3 - −0.25 < m ≤ −0.05     LLM 略低于最优
  2 - −0.05 < m < 0.05      双方势均力敌
  1 - m ≥ 0.05  或  draw_rate ≥ 0.90   LLM 已达/超过最优（饱和）

Optimal 对手策略：
  - 如果提供了 programmatic optimal opponent，直接使用
  - 否则，使用 LLM-as-Optimal 策略：
    - 优先使用 optimal_llm_client（可配置更强的模型）
    - 未配置时回退到 llm_client + expert prompt
"""

import re
from typing import Dict, List, Optional, Callable

# Thinking-mode → (include_thinking, output_format_style)
_THINKING_MODE_CONFIG = {
    "thinking":         {"include_thinking": False, "output_format_style": "simple"},
    "non_thinking_pgt": {"include_thinking": True,  "output_format_style": "thinking_aware"},
    "non_thinking":     {"include_thinking": False, "output_format_style": "simple"},
}


_OPTIMAL_PROMPTS = {
    "KuhnPoker-v0": (
        "You are an expert Kuhn Poker player. Apply GTO (Nash Equilibrium) "
        "strategy strictly:\n"
        "1. CARDS: J < Q < K. With J, you have the worst card. With K, the best.\n"
        "2. BET/BLUFF: With K, always bet. With Q, bet (semi-bluff with showdown value).\n"
        "3. CHECK/FOLD: With J, check if opening, fold if facing a bet. J is "
        "rarely a profitable bluff at equilibrium.\n"
        "4. FACING A BET: Call with K or Q (positive expected value). Fold with J.\n"
        "5. INFORMATION: After the third card is dealt, opponents' ranges are "
        "tighter. Adjust: with K always bet, with J never bluff into a bettor.\n"
        "6. THINK: Always consider what your opponent's range looks like. The "
        "GTO play dominates any exploitative deviation in expectation."
    ),
    "LiarsDice-v0": (
        "You are an expert Liar's Dice player. Apply probability-based strategy:\n"
        "1. COUNT: Tally your own dice by face value. This is your anchor.\n"
        "2. P-HONEST: P(opponent contributes N of face F) is Binomial(opp_dice_remaining, 1/6).\n"
        "3. CALL (Liar): Call when the bid is statistically unlikely — typically "
        "when P(actual ≥ bid) < 30%. Erring on the side of Call protects your dice.\n"
        "4. RAISE: When you have a strong face match, raise quantity (cheaper for your range).\n"
        "5. BLUFF: Bluff with faces you have NONE of only in early rounds. In late "
        "rounds with few dice, never bluff — your face distribution is too tight.\n"
        "6. THINK: Always estimate the joint probability across both players' dice "
        "before calling. A bid of 5 of face 1 with 5 dice each has only ~3% truth."
    ),
    "Othello-v0": (
        "You are an expert Othello (Reversi) player. Follow these strategic principles strictly:\n"
        "1. CORNERS: Corners are the most valuable squares. Prioritize taking corners.\n"
        "2. MOBILITY: Minimize opponent's legal moves while maximizing your own.\n"
        "3. STABILITY: Build stable discs (especially on edges connected to corners).\n"
        "4. PARITY: Control the parity of moves in the endgame.\n"
        "5. AVOID: Never play X-squares (diagonally adjacent to corners) or C-squares "
        "(orthogonally adjacent to corners) unless forced.\n"
        "6. EDGE: Before taking an edge square, ensure it doesn't give opponent a corner.\n"
        "7. EARLY: In the opening, prefer moves that flip few discs.\n"
        "8. THINK: Consider 2-3 moves ahead. Always look for the move that gives you "
        "the best long-term position, not just the most flips."
    ),
    "Breakthrough-v0": (
        "You are an expert Breakthrough player. Follow these strategic principles strictly:\n"
        "1. ADVANCE: Always advance your pawns. Every move should move a pawn forward.\n"
        "2. CHAIN: Keep pawns in a connected diagonal chain so they protect each other.\n"
        "3. BREAKTHROUGH: When a pawn is ahead of opponent's line, push it relentlessly.\n"
        "4. WING: Attack on the wing where opponent has fewer defenders.\n"
        "5. DEFENSE: Never leave your back rank undefended.\n"
        "6. TEMPO: Create multiple threats that force opponent to react.\n"
        "7. SACRIFICE: If sacrificing a pawn creates a breakthrough, do it.\n"
        "8. THINK: Look 3-4 moves ahead. Consider all possible responses."
    ),
    "UltimateTicTacToe-v0": (
        "You are an expert Ultimate Tic Tac Toe player. Follow these principles strictly:\n"
        "1. WIN MICRO: When you can win a micro-board immediately, do it.\n"
        "2. SENTE: Prefer moves that send opponent to micro-boards where they have no advantage.\n"
        "3. CONTROL: Win the center micro-board [4] — it controls the macro-center.\n"
        "4. FORCE: Send opponent to already-won or fully-drawn micro-boards.\n"
        "5. GLOBAL: Think about the macro-board, not just the current micro-board.\n"
        "6. AVOID: Don't give opponent easy wins on micro-boards they already control.\n"
        "7. THINK: Consider how each move affects both the micro and macro board."
    ),
}


class NonSaturationEvaluator:
    """Evaluates the non-saturation dimension of a game."""

    name = "non_saturation"
    weight = 0.35

    def __init__(
        self,
        num_games_vs_optimal: int = 10,
    ):
        self.num_games_vs_optimal = num_games_vs_optimal

    def evaluate(
        self,
        env_id: str,
        llm_client=None,
        random_opponent=None,
        optimal_opponent=None,
        game_wrapper=None,
        optimal_llm_client=None,
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
                "evidence": "SKIPPED: No LLM client provided.",
                "skipped": True,
            }

        from langpsro.utils.prompts import build_agent_prompt
        from langpsro.envs.textarena_env import parse_action

        def make_agent(strategy_prompt, client=None, tag: Optional[str] = None):
            """Build an agent function. Uses specified client or falls back to llm_client."""
            _client = client or llm_client
            _thinking_cfg = _THINKING_MODE_CONFIG.get(thinking_mode, _THINKING_MODE_CONFIG["thinking"])
            def agent(obs):
                prompt_text = build_agent_prompt(
                    strategy_prompt=strategy_prompt,
                    observation=obs,
                    env_id=env_id,
                    include_thinking=_thinking_cfg["include_thinking"],
                    output_format_style=_thinking_cfg["output_format_style"],
                )
                # Tag the call for token-usage accounting (if the client supports it)
                try:
                    raw = _client(prompt_text, tag=tag) if tag else _client(prompt_text)

                    # print('#'*50)
                    # print(prompt_text)
                    # print()
                    # print(raw)
                    # print('#'*50)

                except TypeError:
                    raw = _client(prompt_text)
                return parse_action(raw, env_id, obs)
            return agent

        initial_prompt = (
            f"You are a {env_id} player. Your goal is to win the game. "
            f"Play smart moves based on the current board state."
        )
        # Tag llm_agent's calls so we can measure cost from these rollouts only.
        rollout_tag = "screening:non_saturation:llm"
        llm_agent = make_agent(initial_prompt, tag=rollout_tag)

        # === LLM vs Optimal ===
        optimal_source = "programmatic"
        if optimal_opponent is not None:
            def opt_agent(obs):
                legal = self._extract_legal_actions(obs, env_id)

                # print('#'*50)
                # print(legal)
                # print('#'*50)

                return optimal_opponent.select_action(obs, legal)
        else:
            # LLM-as-Optimal fallback
            # Priority: optimal_llm_client > llm_client + expert prompt
            optimal_client = optimal_llm_client or llm_client
            optimal_prompt = _OPTIMAL_PROMPTS.get(env_id)

            if optimal_prompt is None:
                return {
                    "dimension": self.name,
                    "weight": self.weight,
                    "score": 0,
                    "weighted_score": 0.0,
                    "raw_metrics": {},
                    "evidence": f"SKIPPED: No optimal opponent or expert prompt for {env_id}.",
                    "skipped": True,
                }

            if optimal_llm_client is not None:
                optimal_source = "optimal_llm_client"
                print(f"  [Info] Using optimal LLM client for {env_id}")
            else:
                optimal_source = "llm_as_optimal"
                print(f"  [Info] Using LLM-as-Optimal (same model + expert prompt) for {env_id}")

            opt_agent = make_agent(optimal_prompt, client=optimal_client, tag=None)

        # Capture token usage before the rollout (so we can diff after).
        usage_before = self._get_tag_usage(llm_client, rollout_tag)

        result = game_wrapper.play_multiple_games(
            llm_agent, opt_agent,
            num_games=self.num_games_vs_optimal,
            swap_first=True,
            recorder=recorder,
            matchup_label="llm_vs_optimal",
            agent_labels=("llm", optimal_source),
        )

        usage_after = self._get_tag_usage(llm_client, rollout_tag)

        # LLM is agent_p0, Optimal is agent_p1. swap_first gives both sides
        # the first move, so this is a fair head-to-head.
        llm_wins = result["p0_wins"]
        optimal_wins = result["p1_wins"]
        draws = result["draws"]
        num_games = result["total_games"]
        draw_rate = result["draw_rate"]

        # 胜场差 (win margin): LLM − Optimal. Normalised to [−1, +1].
        margin = llm_wins - optimal_wins
        margin_rate = margin / num_games if num_games > 0 else 0.0
        score = self._compute_score(margin_rate, draw_rate)

        # === Real cost data (steps + tokens) for CostEvaluator ===
        avg_game_length = result.get("avg_game_length", 0.0)
        usage_diff = self._diff_usage(usage_before, usage_after)
        total_tokens = usage_diff["total_tokens"]
        tokens_per_game = total_tokens / num_games if num_games > 0 else 0.0
        llm_calls = usage_diff["call_count"]
        tokens_per_call = total_tokens / llm_calls if llm_calls > 0 else 0.0

        return {
            "dimension": self.name,
            "weight": self.weight,
            "score": score,
            "weighted_score": score * self.weight,
            "raw_metrics": {
                "win_count": llm_wins,
                "loss_count": optimal_wins,
                "draw_count": draws,
                "num_games": num_games,
                "win_rate": result["p0_win_rate"],
                "loss_rate": result["p1_win_rate"],
                "draw_rate": draw_rate,
                "margin": margin,
                "margin_rate": round(margin_rate, 3),
                "optimal_source": optimal_source,
                "source": "llm_games",
                # Cost-related metrics (consumed by CostEvaluator)
                "avg_game_length": round(avg_game_length, 2),
                "total_tokens": total_tokens,
                "tokens_per_game": round(tokens_per_game, 1),
                "tokens_per_call": round(tokens_per_call, 1),
                "llm_calls": llm_calls,
                "token_tracking_supported": total_tokens > 0,
            },
            "evidence": (
                f"LLM vs Optimal ({optimal_source}): "
                f"{llm_wins}胜 {optimal_wins}负 {draws}平, "
                f"胜场差 {margin:+d} (margin_rate {margin_rate:+.0%}). "
                f"评分: {score}/5."
            ),
            "skipped": False,
        }

    @staticmethod
    def _compute_score(margin_rate: float, draw_rate: float = 0.0) -> int:
        """Score non-saturation from the LLM-vs-Optimal win margin.

        margin_rate = (LLM_wins - Optimal_wins) / N, range [−1, +1].
        Strongly negative ⇒ LLM far below optimal ⇒ high non-saturation.
        ~0 / near-all-draws ⇒ LLM at optimal ⇒ saturated.
        """
        if draw_rate >= 0.90:
            return 1  # perfect-play saturation (e.g. TTT optimal vs optimal)
        if margin_rate <= -0.50:
            return 5
        if margin_rate <= -0.25:
            return 4
        if margin_rate <= -0.05:
            return 3
        if margin_rate < 0.05:
            return 2
        return 1

    @staticmethod
    def _get_tag_usage(client, tag: str) -> Dict:
        """Snapshot a tagged client's cumulative token usage + call count.

        Returns a dict with prompt/completion/total token counts and the
        number of calls so far for the given tag. If the client doesn't
        expose usage tracking we return zeros — the cost evaluator falls
        back to step-based estimates in that case.
        """
        zero = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }
        if client is None:
            return zero
        try:
            usage = client.get_usage(tag=tag)
            call_log = getattr(client, "get_call_log", lambda: [])()
            call_count = sum(1 for entry in call_log if entry.get("tag") == tag)
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                "call_count": call_count,
            }
        except Exception:
            return zero

    @staticmethod
    def _diff_usage(before: Dict, after: Dict) -> Dict:
        """Subtract two usage snapshots produced by _get_tag_usage."""
        return {
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "call_count": after["call_count"] - before["call_count"],
        }

    @staticmethod
    def _extract_legal_actions(observation: str, env_id: str):
        """Extract legal actions from a TextArena observation.

        Dispatches by env_id since each game has a different action
        vocabulary. When the obs doesn't yield a parseable action set,
        we fall back to a per-env default that lets the LLM produce
        *some* legal move (so the rollout doesn't crash).
        """
        import re
        if env_id == "KuhnPoker-v0":
            return _extract_legal_kuhnpoker(observation)
        if env_id == "LiarsDice-v0":
            return _extract_legal_liarsdice(observation)
        if env_id == "Othello-v0":
            return _extract_legal_othello(observation)
        if env_id == "Breakthrough-v0":
            return _extract_legal_breakthrough(observation)
        # Default: numeric "[N]" actions (TicTacToe / ConnectFour / Nim).
        actions = re.findall(r'\[(\d+)\]', observation)
        seen = set()
        result = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                result.append(f"[{a}]")
        if not result:
            return [f"[{i}]" for i in range(9)]
        return result


# ---------------------------------------------------------------------------
# Per-game legal-action extractors (used by _extract_legal_actions above)
# ---------------------------------------------------------------------------

# Kuhn Poker: actions are textual tokens "[check]"/"[bet]"/"[call]"/"[fold]".
# TextArena publishes them on a line like
#   "Your available actions are: '[check]', '[bet]'"
#
# Cumulative-obs trap: the previous round's "Your available actions" line
# is still visible. We must take the LAST such line, AND it must be the
# one for the CURRENT (live) round. A "live round" boundary in TextArena's
# KuhnPoker obs is the "### Starting round N" marker; we use the last
# `### Starting round` occurrence as the anchor and only consider lines
# that follow it.
_KUHN_TOK_RE = re.compile(r"\[(check|bet|call|fold)\]")


def _extract_legal_kuhnpoker(observation: str) -> List[str]:
    # Find the position of the LAST "### Starting round" marker; the
    # current "Your available actions" line is the one closest to (and
    # after) that marker. If no such marker exists, fall back to the
    # very last "Your available actions" line in the obs.
    last_round_idx = observation.rfind("### Starting round")
    search_window = observation[last_round_idx:] if last_round_idx >= 0 else observation
    avail_line = ""
    for line in search_window.splitlines()[::-1]:
        if "Your available actions" in line:
            avail_line = line
            break
    if avail_line:
        out: List[str] = []
        seen = set()
        for tok in _KUHN_TOK_RE.findall(avail_line):
            full = f"[{tok}]"
            if full not in seen:
                seen.add(full)
                out.append(full)
        if out:
            return out
    # Conservative default: the opening actions only.
    return ["[check]", "[bet]"]


# Liar's Dice: actions are "[Bid: <qty>, <face>]" or "[Call]".
# We parse the current bid (if any) from the latest "bids N of face F."
# line and enumerate legal raises plus [Call].
_LIAR_BID_LINE_RE = re.compile(r"bids\s+(\d+)\s+of\s+face\s+(\d+)\.")


def _extract_legal_liarsdice(observation: str) -> List[str]:
    out: List[str] = []
    seen = set()

    # Always advertise [Call] — env generally accepts it whenever a
    # bid is on the table.
    for tok in ("[Call]",):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)

    # Extract the most recent bid (qty, face).
    cur_q, cur_f = 0, 0
    matches = _LIAR_BID_LINE_RE.findall(observation)
    if matches:
        try:
            cur_q, cur_f = int(matches[-1][0]), int(matches[-1][1])
        except (ValueError, IndexError):
            cur_q = cur_f = 0

    # Enumerate legal raises: q' > q OR f' > f, q'<=12, f'<=6.
    max_q = max(cur_q + 5, 6)
    for q in range(1, max_q + 1):
        for f in range(1, 7):
            if q > cur_q or f > cur_f:
                cand = f"[Bid: {q}, {f}]"
                if cand not in seen:
                    seen.add(cand)
                    out.append(cand)
    return out


# Othello: actions are "[row, col]" e.g. "[2, 3]".
# The env publishes them in a "Valid moves:" line.
_OTHELLO_MOVE_RE = re.compile(r"\[(\d+),\s*(\d+)\]")


def _extract_legal_othello(observation: str) -> List[str]:
    """Extract legal Othello moves from the last 'Valid moves:' line."""
    idx = observation.rfind("Valid moves:")
    if idx >= 0:
        out: List[str] = []
        seen = set()
        for r, c in _OTHELLO_MOVE_RE.findall(observation[idx:]):
            a = f"[{r}, {c}]"
            if a not in seen:
                seen.add(a)
                out.append(a)
        if out:
            return out
    return ["[2, 3]", "[3, 2]", "[4, 5]", "[5, 4]"]


# Breakthrough: actions are UCI-like "[a2a3]".
_BT_MOVE_RE = re.compile(r"\[([a-h][1-8][a-h][1-8])\]")


def _extract_legal_breakthrough(observation: str) -> List[str]:
    """Extract all UCI-format moves from the obs.

    In the initial obs the only matching tokens are in the rules text
    ('[a2a3]' example), which gives 1 item.  After the first move the
    obs contains the move history too, so we get more.  The env does
    NOT enumerate legal moves for Breakthrough — we rely on the state
    representation for accuracy.  This extractor is a best-effort that
    at least avoids crashing the rollout.
    """
    out: List[str] = []
    seen = set()
    for m in _BT_MOVE_RE.findall(observation):
        full = f"[{m}]"
        if full not in seen:
            seen.add(full)
            out.append(full)
    if not out:
        return ["[a2a3]"]
    return out
