"""
TicTacToe 游戏状态 (3×3, 先手 X = Player 1, 后手 O = Player 0).

TextArena TextArena 约定:
  - Player 0 plays 'O', Player 1 plays 'X'
  - Player 1 (X) moves first
  - Action: [0]–[8]
  - Observation 是累积式, 但 parser 只取最后一个棋盘。
"""

import re
from typing import List, Optional, Tuple

# 行列连线
_LINES = (
    (0,1,2), (3,4,5), (6,7,8),  # 行
    (0,3,6), (1,4,7), (2,5,8),  # 列
    (0,4,8), (2,4,6),            # 对角
)

_PLAYER_RE = re.compile(r"You are Player (\d+)")
_BOARD_RE = re.compile(r"^\s*([XO\d])\s*\|", re.MULTILINE)


class TicTacToeState:
    """Immutable TicTacToe 状态。"""

    __slots__ = ("_cells", "_player_to_move")

    def __init__(self, cells: Tuple[str, ...], player_to_move: int):
        self._cells = cells          # 9-length tuple: '' / 'X' / 'O'
        self._player_to_move = player_to_move  # 0 or 1

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def new_game(cls) -> "TicTacToeState":
        cells = tuple("" for _ in range(9))
        return cls(cells, player_to_move=1)  # X 先手

    @classmethod
    def parse_from_obs(cls, observation: str, my_player_id: int) -> "TicTacToeState":
        """从 TextArena obs 解析当前棋盘。
        obs 是累积式的, 我们取最后一个三行棋盘块。
        """
        # 找最后的 "Current Board:" 之后的三行
        idx = observation.rfind("Current Board:")
        if idx < 0:
            return cls.new_game()
        tail = observation[idx:]
        lines = tail.splitlines()
        cells = [""] * 9
        ci = 0
        for line in lines:
            m = _BOARD_RE.match(line)
            if not m:
                continue
            # 行格式: " O | X |  "  或 " 0 | 1 | 2 "
            tokens = re.findall(r"[XO\d]", line)
            for tok in tokens[:3]:
                if tok in ("X", "O"):
                    cells[ci] = tok
                else:
                    cells[ci] = ""
                ci += 1
            if ci >= 9:
                break

        # 推断轮到谁
        # X 的个数 = player 1 的步数
        x_count = sum(1 for c in cells if c == "X")
        o_count = sum(1 for c in cells if c == "O")
        # X 先手, 所以 X == O or X == O+1
        player_to_move = 1 if x_count == o_count else 0

        return cls(tuple(cells), player_to_move)

    # ------------------------------------------------------------------
    # 搜索协议
    # ------------------------------------------------------------------
    @property
    def player_to_move(self) -> int:
        return self._player_to_move

    def get_legal_actions(self) -> List[str]:
        return [f"[{i}]" for i, c in enumerate(self._cells) if c == ""]

    def apply_action(self, action: str) -> "TicTacToeState":
        m = re.match(r"\[(\d+)\]", action)
        if not m:
            return self
        idx = int(m.group(1))
        if self._cells[idx] != "":
            return self
        sym = "X" if self._player_to_move == 1 else "O"
        new_cells = list(self._cells)
        new_cells[idx] = sym
        return TicTacToeState(tuple(new_cells), 1 - self._player_to_move)

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        # 检查连线
        for a, b, c in _LINES:
            if self._cells[a] and self._cells[a] == self._cells[b] == self._cells[c]:
                winner = 1 if self._cells[a] == "X" else 0
                return True, winner
        if all(self._cells):
            return True, None  # draw
        return False, None

    def evaluate(self, player: int) -> float:
        """简单启发: 潜在连线数差值."""
        if self.is_terminal()[0]:
            _, winner = self.is_terminal()
            if winner is None:
                return 0.0
            return 1e6 if winner == player else -1e6
        return self._line_score(player) - self._line_score(1 - player)

    def _line_score(self, player: int) -> float:
        sym = "X" if player == 1 else "O"
        score = 0.0
        for a, b, c in _LINES:
            line = [self._cells[a], self._cells[b], self._cells[c]]
            my = line.count(sym)
            opp = line.count("X" if sym == "O" else "O")
            if my == 2 and opp == 0:
                score += 10.0
            elif my == 1 and opp == 0:
                score += 1.0
            elif my == 1 and opp == 1:
                score += 0.0
            # 对方已有 2 个但空了 1 个 → 威胁 (负值)
            if opp == 2 and my == 0:
                score -= 5.0
        return score

    def move_key(self, action: str) -> int:
        """Move ordering: 中心 [4] > 角 [0,2,6,8] > 边 [1,3,5,7]."""
        idx = int(action.strip("[]"))
        order = {4: 2, 0: 1, 2: 1, 6: 1, 8: 1}
        return order.get(idx, 0)

    # ------------------------------------------------------------------
    # Hash / eq (转置表用)
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash(self._cells)

    def __eq__(self, other) -> bool:
        return isinstance(other, TicTacToeState) and self._cells == other._cells

    def __repr__(self) -> str:
        rows = []
        for r in range(3):
            row = " ".join(self._cells[r*3:(r+1)*3] or "." for _ in range(3))
            rows.append(row)
        return "TTT(" + " | ".join(rows) + f") P={self._player_to_move}"
