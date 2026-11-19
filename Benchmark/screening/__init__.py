"""
Game Screening Module for Self-Play Self-Evolving Benchmark

Systematically evaluates zero-sum games across three dimensions to determine
their suitability for LLM Self-Play Self-Evolving research:

    1. Non-saturation (35%)  — LLM 初始表现与最优水平的差距
    2. Evolvability (55%)    — 策略改进空间（含可测量性）
    3. Cost efficiency (10%) — 运行成本（基于真实 token / 步数）

Usage:
    from screening import GameScreener
    screener = GameScreener(llm_client)
    report = screener.screen("TicTacToe-v0")
    print(report.summary())
"""

from .core import GameScreener, ScreeningReport, ScreeningDimension

__all__ = ["GameScreener", "ScreeningReport", "ScreeningDimension"]
