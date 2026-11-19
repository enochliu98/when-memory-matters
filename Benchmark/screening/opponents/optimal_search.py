"""
通用搜索引擎封装器: 把 screening/search/ 包成 BaseOpponent 接口.

用于 TTT / C4 / Breakthrough / Othello 等没有闭式最优解的游戏.
有解析最优解的游戏 (Nim / KuhnPoker / LiarsDice) 不走这里.

每个游戏配置:
- TTT:         minimax d=9 (完美走法, 0 失败)
- C4:          minimax d=4 (100% vs random)
- Breakthrough: minimax d=3 (96%+ vs random; d=4 有深度倒置)
- Othello:     MCTS 150 iter (~90% vs random; minimax 评估函数信噪比过低)
"""

from typing import List

from ..search import minimax, mcts
from ..search.games import (
    BreakthroughState, ConnectFourState, OthelloState, TicTacToeState,
)

from .base import BaseOpponent


# (StateCls, engine_name, engine_kwargs)
_SEARCH_CONFIG = {
    "TicTacToe-v0":    (TicTacToeState,   "minimax", {"depth": 9, "use_tt": True}),
    "ConnectFour-v0":  (ConnectFourState, "minimax", {"depth": 4, "use_tt": True}),
    "Breakthrough-v0": (BreakthroughState,"minimax", {"depth": 3, "use_tt": True}),
    "Othello-v0":      (OthelloState,     "mcts",    {"iterations": 150, "c": 1.4}),
}


class SearchOptimalOpponent(BaseOpponent):
    """把通用搜索算法 (minimax / mcts) 包成 BaseOpponent 接口.

    `select_action(obs, legal_actions)` 用 `parse_from_obs(obs, 0)` 解析状态,
    然后调用 `minimax_move` 或 `mcts_move` 选一步, 最后从 `legal_actions`
    中过滤 (理论上应一致, 但保险起见过滤一次).
    """

    def __init__(self, env_id: str):
        if env_id not in _SEARCH_CONFIG:
            raise ValueError(
                f"No search config for env {env_id!r}. "
                f"Supported: {list(_SEARCH_CONFIG)}"
            )
        StateCls, engine, kwargs = _SEARCH_CONFIG[env_id]
        self.StateCls = StateCls
        self._engine = engine
        self._kwargs = kwargs
        super().__init__(f"search_{engine}_{env_id[:-3]}")

    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        if not legal_actions:
            return ""
        state = self.StateCls.parse_from_obs(observation, 0)
        if self._engine == "minimax":
            return minimax.minimax_move(state, **self._kwargs)
        else:
            return mcts.mcts_move(state, **self._kwargs)
