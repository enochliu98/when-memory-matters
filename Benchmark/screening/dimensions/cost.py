"""
Cost Evaluator (权重 10%)

从 NonSaturationEvaluator 的实测对局数据中读取真实成本指标：
  - 平均每局步数（来自 game_wrapper.play_multiple_games 的 avg_game_length）
  - 平均每局 token 消耗（来自带 tag 的 LLM client.get_usage(tag)）

评分逻辑（双指标取下限，token 优先；token 不可用时回退到步数）：

  Tokens-per-game (优先)：
    5 - < 2 000        极低
    4 - 2 000 - 8 000  低
    3 - 8 000 - 30 000 中等
    2 - 30 000 - 100 000 高
    1 - > 100 000       极高

  Steps-per-game (回退)：
    5 - < 5
    4 - 5  - 15
    3 - 15 - 50
    2 - 50 - 200
    1 - > 200
"""

from typing import Dict, Optional


# Token thresholds in tokens-per-game
_TOKEN_TIERS = [
    (2_000,    5),
    (8_000,    4),
    (30_000,   3),
    (100_000,  2),
]

# Step thresholds in steps-per-game (used when token data is unavailable)
_STEP_TIERS = [
    (5,   5),
    (15,  4),
    (50,  3),
    (200, 2),
]


def _score_by_tiers(value: float, tiers) -> int:
    for upper, score in tiers:
        if value < upper:
            return score
    return 1


class CostEvaluator:
    """Evaluates the cost dimension using actual rollout data.

    Pulls real per-game step counts and (when available) per-game token
    counts from the NonSaturationEvaluator's rollouts. No hardcoded
    per-game heuristics — if rollouts haven't run, the dimension is
    skipped.
    """

    name = "cost"
    weight = 0.10

    def evaluate(
        self,
        env_id: str,
        non_saturation_results: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        del env_id  # accepted for signature consistency; cost is purely data-driven

        if non_saturation_results is None or non_saturation_results.get("skipped", True):
            return {
                "dimension": self.name,
                "weight": self.weight,
                "score": 0,
                "weighted_score": 0.0,
                "raw_metrics": {},
                "evidence": "SKIPPED: No game data available (non_saturation rollouts didn't run).",
                "skipped": True,
            }

        raw = non_saturation_results.get("raw_metrics", {})
        avg_steps = raw.get("avg_game_length", 0.0)
        tokens_per_game = raw.get("tokens_per_game", 0.0)
        tokens_per_call = raw.get("tokens_per_call", 0.0)
        total_tokens = raw.get("total_tokens", 0)
        token_supported = raw.get("token_tracking_supported", False)
        llm_calls = raw.get("llm_calls", 0)

        # Decide scoring source: prefer real tokens; fall back to step count.
        if token_supported and tokens_per_game > 0:
            score = _score_by_tiers(tokens_per_game, _TOKEN_TIERS)
            source = "tokens_per_game"
            evidence = (
                f"实测每局 ≈ {avg_steps:.1f} 步, "
                f"LLM token/局 = {tokens_per_game:,.0f} "
                f"({tokens_per_call:,.0f}/call × {llm_calls} calls). "
                f"评分: {score}/5 (基于 token)."
            )
        elif avg_steps > 0:
            score = _score_by_tiers(avg_steps, _STEP_TIERS)
            source = "avg_game_length"
            evidence = (
                f"实测每局 ≈ {avg_steps:.1f} 步 (token 追踪不可用). "
                f"评分: {score}/5 (基于步数)."
            )
        else:
            return {
                "dimension": self.name,
                "weight": self.weight,
                "score": 0,
                "weighted_score": 0.0,
                "raw_metrics": {},
                "evidence": "SKIPPED: No usable trajectory data (steps=0 and tokens=0).",
                "skipped": True,
            }

        return {
            "dimension": self.name,
            "weight": self.weight,
            "score": score,
            "weighted_score": score * self.weight,
            "raw_metrics": {
                "avg_game_length": avg_steps,
                "tokens_per_game": tokens_per_game,
                "tokens_per_call": tokens_per_call,
                "total_tokens": total_tokens,
                "llm_calls": llm_calls,
                "score_source": source,
            },
            "evidence": evidence,
            "skipped": False,
        }
