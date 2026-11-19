"""
Core GameScreener — orchestrates the three-dimensional screening.

Dimensions:
  1. Non-saturation (35%) — LLM vs Random + vs Optimal
  2. Evolvability (55%) — naive prompt vs expert prompt (含可测量性)
  3. Cost (10%) — actual game length & token consumption from rollouts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .dimensions import (
    NonSaturationEvaluator,
    EvolvabilityEvaluator,
    CostEvaluator,
)
from .recorder import GameRecorder


@dataclass
class ScreeningDimension:
    name: str
    weight: float
    score: int
    weighted_score: float
    raw_metrics: Dict
    evidence: str
    skipped: bool = False


@dataclass
class ScreeningReport:
    env_id: str
    timestamp: str
    dimensions: Dict[str, ScreeningDimension]
    total_score: float
    config: Dict
    game_records: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Game Screening Report: {self.env_id} ===",
            f"Total Score: {self.total_score:.1f}/5.0",
            f"Timestamp: {self.timestamp}",
            "",
            "Dimension Breakdown:",
        ]
        for dim_name in ["non_saturation", "evolvability", "cost"]:
            d = self.dimensions.get(dim_name)
            if d:
                if d.skipped:
                    lines.append(f"  {d.name:20s} SKIPPED  (no LLM client)")
                else:
                    bar = "█" * d.score + "░" * (5 - d.score)
                    lines.append(
                        f"  {d.name:20s} {d.score}/5  {bar}  "
                        f"(w={d.weight:.0%}, contrib={d.weighted_score:.2f})"
                    )
                    lines.append(f"    {d.evidence}")

        lines.append("")
        if self.total_score >= 4.0:
            lines.append("Verdict: ✅ STRONGLY RECOMMENDED for Self-Play Self-Evolving")
        elif self.total_score >= 3.0:
            lines.append("Verdict: ⚠️ RECOMMENDED (consider limitations)")
        elif self.total_score >= 2.0:
            lines.append("Verdict: ❌ NOT RECOMMENDED (significant limitations)")
        else:
            lines.append("Verdict: 🚫 AVOID (saturated / unsuitable)")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "env_id": self.env_id,
            "timestamp": self.timestamp,
            "total_score": self.total_score,
            "config": self.config,
            "game_record_count": len(self.game_records),
            "dimensions": {
                name: {
                    "score": d.score,
                    "weight": d.weight,
                    "weighted_score": d.weighted_score,
                    "raw_metrics": d.raw_metrics,
                    "evidence": d.evidence,
                    "skipped": d.skipped,
                }
                for name, d in self.dimensions.items()
            },
        }


class GameScreener:
    """
    Orchestrates the three-dimensional screening for a zero-sum game.
    """

    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        optimal_llm_client: Optional[Callable] = None,
        num_games_vs_optimal: int = 10,
        num_evolvability_games: int = 10,
        record_games: bool = True,
        thinking_mode: str = "thinking",
    ):
        self.llm_client = llm_client
        self.optimal_llm_client = optimal_llm_client
        self.record_games = record_games
        self.thinking_mode = thinking_mode

        self.evaluators = {
            "non_saturation": NonSaturationEvaluator(
                num_games_vs_optimal=num_games_vs_optimal,
            ),
            "evolvability": EvolvabilityEvaluator(
                num_games=num_evolvability_games,
            ),
            "cost": CostEvaluator(),
        }

    def screen(
        self,
        env_id: str,
        random_opponent=None,
        optimal_opponent=None,
        game_wrapper=None,
        optimal_llm_client=None,
    ) -> ScreeningReport:
        # Allow per-call override of optimal_llm_client
        eff_optimal_llm = optimal_llm_client or self.optimal_llm_client

        config = {
            "llm_provided": self.llm_client is not None,
            "model": getattr(self.llm_client, "model", None),
            "reference_model": getattr(eff_optimal_llm, "model", None),
            "optimal_llm_provided": eff_optimal_llm is not None,
            "record_games": self.record_games,
            "thinking_mode": self.thinking_mode,
        }

        # One recorder per screen() call; threaded through both evaluators so
        # every game (vs random, vs optimal, evolvability baseline/expert) is
        # captured with its matchup tag.
        recorder = GameRecorder() if self.record_games else None

        shared_kwargs = {
            "env_id": env_id,
            "llm_client": self.llm_client,
            "optimal_llm_client": eff_optimal_llm,
            "random_opponent": random_opponent,
            "optimal_opponent": optimal_opponent,
            "game_wrapper": game_wrapper,
            "recorder": recorder,
            "thinking_mode": self.thinking_mode,
        }

        results = {}

        # 1. Non-saturation (run first — cost depends on its rollout data)
        results["non_saturation"] = self.evaluators["non_saturation"].evaluate(**shared_kwargs)

        # 2. Evolvability (independent: Expert vs Naive head-to-head)
        results["evolvability"] = self.evaluators["evolvability"].evaluate(**shared_kwargs)

        # 3. Cost (real avg_game_length + token usage from non_saturation rollouts)
        results["cost"] = self.evaluators["cost"].evaluate(
            env_id=env_id,
            non_saturation_results=results["non_saturation"],
        )

        dimensions = {}
        for name, result in results.items():
            dimensions[name] = ScreeningDimension(
                name=name,
                weight=result.get("weight", 0),
                score=result.get("score", 0),
                weighted_score=result.get("weighted_score", 0.0),
                raw_metrics=result.get("raw_metrics", {}),
                evidence=result.get("evidence", ""),
                skipped=result.get("skipped", False),
            )

        total_score = sum(d.weighted_score for d in dimensions.values())

        return ScreeningReport(
            env_id=env_id,
            timestamp=datetime.now().isoformat(),
            dimensions=dimensions,
            total_score=round(total_score, 1),
            config=config,
            game_records=list(recorder.records) if recorder is not None else [],
        )

    def screen_multiple(
        self,
        env_ids: List[str],
        **kwargs,
    ) -> Dict[str, ScreeningReport]:
        return {
            env_id: self.screen(env_id, **kwargs)
            for env_id in env_ids
        }
