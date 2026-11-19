"""
ConnectFour 游戏状态 (6 行 × 7 列, row 0 = top, row 5 = bottom).

TextArena 约定:
  - Player 0 = X (先手), Player 1 = O
  - Action: [0]–[6]
  - 棋子沉到 column 底部 (最大 row index)
"""

import re
from typing import List, Optional, Tuple

ROWS, COLS = 6, 7

# 中心优先列序
_COL_ORDER = (3, 2, 4, 1, 5, 0, 6)

# 位置权重 (row × col; row 0 = top)
_POS_WEIGHTS = (
    (3, 4, 5, 7, 5, 4, 3),
    (4, 6, 8, 10, 8, 6, 4),
    (5, 8, 11, 13, 11, 8, 5),
    (5, 8, 11, 13, 11, 8, 5),
    (4, 6, 8, 10, 8, 6, 4),
    (3, 4, 5, 7, 5, 4, 3),
)

_PLAYER_RE = re.compile(r"You are Player (\d+)")
_BOARD_HEADER_RE = re.compile(r"^\s*0\s+1\s+2\s+3\s+4\s+5\s+6\s*$")
_BOARD_ROW_RE = re.compile(r"^([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s*$")


class ConnectFourState:
    """Immutable ConnectFour 状态。board 以 tuple[tuple[str,...],...] 存储, row 0 = top。"""

    __slots__ = ("board", "_player_to_move")

    def __init__(self, board: Tuple[Tuple[str, ...], ...], player_to_move: int):
        self.board = board
        self._player_to_move = player_to_move

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------
    @classmethod
    def new_game(cls) -> "ConnectFourState":
        board = tuple(tuple("." for _ in range(COLS)) for _ in range(ROWS))
        return cls(board, player_to_move=0)  # Player 0 (X) 先手

    @classmethod
    def parse_from_obs(cls, observation: str, my_player_id: int) -> "ConnectFourState":
        """从 TextArena obs 解析最后的 6×7 棋盘。"""
        lines = observation.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if not _BOARD_HEADER_RE.match(lines[i]):
                continue
            if i + 1 >= len(lines):
                continue
            board_rows: List[Tuple[str, ...]] = []
            for k in range(i + 2, min(i + 2 + ROWS, len(lines))):
                m = _BOARD_ROW_RE.match(lines[k])
                if not m:
                    break
                board_rows.append(tuple(m.groups()))
            if len(board_rows) == ROWS:
                # 推断轮到谁: X 数与 O 数
                flat = [c for row in board_rows for c in row]
                xc = flat.count("X")
                oc = flat.count("O")
                p = 0 if xc == oc else 1
                return cls(tuple(board_rows), p)
        return cls.new_game()

    # ------------------------------------------------------------------
    # 搜索协议
    # ------------------------------------------------------------------
    @property
    def player_to_move(self) -> int:
        return self._player_to_move

    def get_legal_actions(self) -> List[str]:
        return [f"[{c}]" for c in range(COLS) if self.board[0][c] == "."]

    def apply_action(self, action: str) -> "ConnectFourState":
        m = re.match(r"\[(\d+)\]", action)
        if not m:
            return self
        col = int(m.group(1))
        if col < 0 or col >= COLS or self.board[0][col] != ".":
            return self
        sym = _symbol(self._player_to_move)
        new_board = [list(row) for row in self.board]
        for r in range(ROWS - 1, -1, -1):
            if new_board[r][col] == ".":
                new_board[r][col] = sym
                break
        return ConnectFourState(tuple(tuple(r) for r in new_board),
                                1 - self._player_to_move)

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        for r in range(ROWS):
            for c in range(COLS):
                sym = self.board[r][c]
                if sym == ".":
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    count = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr][cc] == sym:
                        count += 1
                        rr += dr
                        cc += dc
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr][cc] == sym:
                        count += 1
                        rr -= dr
                        cc -= dc
                    if count >= 4:
                        winner = 0 if sym == "X" else 1
                        return True, winner
        if all(self.board[0][c] != "." for c in range(COLS)):
            return True, None
        return False, None

    def evaluate(self, player: int) -> float:
        """位置权重 + open-3 威胁."""
        my_sym = "X" if player == 0 else "O"
        opp_sym = "O" if my_sym == "X" else "X"
        score = 0.0
        for r in range(ROWS):
            for c in range(COLS):
                v = self.board[r][c]
                if v == my_sym:
                    score += _POS_WEIGHTS[r][c]
                elif v == opp_sym:
                    score -= _POS_WEIGHTS[r][c]
        # 四元组扫描
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rr, cc = r + 3 * dr, c + 3 * dc
                    if not (0 <= rr < ROWS and 0 <= cc < COLS):
                        continue
                    window = [self.board[r + i * dr][c + i * dc] for i in range(4)]
                    mc = window.count(my_sym)
                    oc = window.count(opp_sym)
                    ec = window.count(".")
                    if oc == 0 and mc == 3 and ec == 1:
                        score += 50.0
                    elif oc == 0 and mc == 2 and ec == 2:
                        score += 5.0
                    elif mc == 0 and oc == 3 and ec == 1:
                        score -= 60.0
                    elif mc == 0 and oc == 2 and ec == 2:
                        score -= 5.0
        return score

    def move_key(self, action: str) -> int:
        col = int(action.strip("[]"))
        return -_COL_ORDER.index(col)  # 中心优先

    # ------------------------------------------------------------------
    # Hash / eq
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash(self.board)

    def __eq__(self, other) -> bool:
        return isinstance(other, ConnectFourState) and self.board == other.board

    def __repr__(self) -> str:
        return f"C4(P{self._player_to_move})"


def _symbol(player: int) -> str:
    return "X" if player == 0 else "O"
