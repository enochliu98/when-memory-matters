"""各游戏状态实现。

每个文件实现一个 GameState 类, 遵循统一的搜索协议：
  - get_legal_actions() → List[str]
  - apply_action(action) → GameState
  - is_terminal() → Tuple[bool, Optional[int]]
  - evaluate(player) → float
  - player_to_move → int
  - move_key(action) → Any          (可选, move ordering)
  - __hash__ / __eq__                (转置表用)
"""

from .tictactoe import TicTacToeState
from .connectfour import ConnectFourState
from .breakthrough import BreakthroughState
from .othello import OthelloState

__all__ = [
    "TicTacToeState",
    "ConnectFourState",
    "BreakthroughState",
    "OthelloState",
]
