"""
通用搜索算法库 (不依赖已有 opponents 文件).

提供：
    screening.search.minimax  — Negamax + α-β 剪枝 + 转置表
    screening.search.mcts     — UCT MCTS
    screening.search.games    — 各游戏状态实现
"""

from . import minimax, mcts, games

__all__ = ["minimax", "mcts", "games"]
