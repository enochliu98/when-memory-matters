"""
Breakthrough 游戏状态 (8×8, 白方向上、黑方向下移动).

TextArena 约定:
  - Player 0 = White (行 1–2, 向上 → 行 8), Player 1 = Black (行 7–8, 向下 → 行 1)
  - Action: `[from_sq to_sq]`, e.g. [a2a3]; 使用 UCI 列标 (a–h) + 数字行 (1–8)
  - 首先到达对方底行者胜; 所有子被吃完者负。

Board 内部表示: 1-D list[64], 索引 = (row-1)*8 + col (row 1=bottom, col 0=a).
值为 'W' / 'B' / ''.
"""

import math, re
from typing import List, Optional, Tuple

COLS = 8
ROWS = 8
SQUARES = 64

_COL_LETTERS = "abcdefgh"

_PLAYER_RE = re.compile(r"You are (White|Black|Player (\d+))")
_SQUARE_RE = re.compile(r"[a-h][1-8]")
_MOVE_RE = re.compile(r"\[([a-h][1-8])([a-h][1-8])\]")


class BreakthroughState:
    """Immutable Breakthrough 状态。"""

    __slots__ = ("_pieces", "_player_to_move")

    def __init__(self, pieces: Tuple[str, ...], player_to_move: int):
        self._pieces = pieces
        self._player_to_move = player_to_move

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------
    @classmethod
    def new_game(cls) -> "BreakthroughState":
        pieces = [""] * SQUARES
        # White rows 1,2 (index 0..7, 8..15)
        for c in range(COLS):
            pieces[c] = "W"           # row 1
            pieces[COLS + c] = "W"    # row 2
        # Black rows 7,8 (index 48..55, 56..63)
        for c in range(COLS):
            pieces[6 * COLS + c] = "B"  # row 7
            pieces[7 * COLS + c] = "B"  # row 8
        return cls(tuple(pieces), player_to_move=0)  # White 先手

    @classmethod
    def parse_from_obs(cls, observation: str, my_player_id: int) -> "BreakthroughState":
        """从 TextArena 累积 obs 解析棋盘。

        TextArena obs 是累积式, 包含每步后的完整棋盘渲染。
        取最后 8 行匹配行作为当前棋盘。正确推断先后手:
        - 查找 obs 中最后一个 "[White]" 或 "[Black]" 标记
          (表示最近一步是谁走的), 反之即该谁走。
        """
        pieces = [""] * SQUARES
        matches = []
        for line in observation.splitlines():
            if "|" not in line:
                continue
            m = re.match(r"\s*\d\s*\|(.+)", line)
            if not m:
                continue
            content = m.group(1).strip()
            toks = content.split()
            row_num = int(line.strip().split("|")[0].strip())
            if row_num < 1 or row_num > 8:
                continue
            idx = 0
            for t in toks:
                if t in ("W", "B"):
                    pieces[(row_num - 1) * COLS + idx] = t
                idx += 1
            matches.append((row_num, content))
        if not matches:
            return cls.new_game()

        # 最后 8 行 match = 当前棋盘块
        pieces = [""] * SQUARES
        for row_num, content in matches[-8:]:
            toks = content.split()
            for c, tok in enumerate(toks[:COLS]):
                if tok in ("W", "B"):
                    pieces[(row_num - 1) * COLS + c] = tok

        # 推断轮到谁: 找 obs 中最后一个 "[White]" 或 "[Black]"
        last_white = observation.rfind("[White]")
        last_black = observation.rfind("[Black]")
        player_to_move = 1 if last_white > last_black else 0
        return cls(tuple(pieces), player_to_move)

    # ------------------------------------------------------------------
    # 搜索协议
    # ------------------------------------------------------------------
    @property
    def player_to_move(self) -> int:
        return self._player_to_move

    def _sq_index(self, col: int, row: int) -> int:
        """row 1..8, col 0..7 (a=0)."""
        return (row - 1) * COLS + col

    def _col_row(self, idx: int) -> Tuple[int, int]:
        return idx % COLS, idx // COLS + 1

    def _coord(self, sq: str) -> int:
        """'a2' → index."""
        col = ord(sq[0]) - ord("a")
        row = int(sq[1])
        return self._sq_index(col, row)

    def _square_str(self, idx: int) -> str:
        c, r = self._col_row(idx)
        return chr(ord("a") + c) + str(r)

    def get_legal_actions(self) -> List[str]:
        actions = []
        my_piece = "W" if self._player_to_move == 0 else "B"
        # 白向上 (row+1), 黑向下 (row-1)
        direction = 1 if my_piece == "W" else -1
        opp_piece = "B" if my_piece == "W" else "W"

        for idx, p in enumerate(self._pieces):
            if p != my_piece:
                continue
            col, row = self._col_row(idx)
            new_row = row + direction
            if new_row < 1 or new_row > 8:
                continue
            nidx = self._sq_index(col, new_row)
            # 直走
            if self._pieces[nidx] == "":
                actions.append(f"[{self._square_str(idx)}{self._square_str(nidx)}]")
            # 斜吃
            for dc in (-1, 1):
                nc = col + dc
                if nc < 0 or nc >= COLS:
                    continue
                diag_idx = self._sq_index(nc, new_row)
                if self._pieces[diag_idx] == opp_piece:
                    actions.append(f"[{self._square_str(idx)}{self._square_str(diag_idx)}]")
        return actions

    def apply_action(self, action: str) -> "BreakthroughState":
        m = _MOVE_RE.match(action)
        if not m:
            return self
        from_sq = self._coord(m.group(1))
        to_sq = self._coord(m.group(2))
        if not (0 <= from_sq < SQUARES and 0 <= to_sq < SQUARES):
            return self
        new_pieces = list(self._pieces)
        piece = new_pieces[from_sq]
        if piece == "":
            return self
        new_pieces[to_sq] = piece
        new_pieces[from_sq] = ""
        return BreakthroughState(tuple(new_pieces), 1 - self._player_to_move)

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        # 检查是否到达对面底线
        for i, p in enumerate(self._pieces):
            _, row = self._col_row(i)
            if p == "W" and row == 8:
                return True, 0
            if p == "B" and row == 1:
                return True, 1
        # 检查是否有棋子全灭
        wc = sum(1 for p in self._pieces if p == "W")
        bc = sum(1 for p in self._pieces if p == "B")
        if wc == 0 and bc > 0:
            return True, 1
        if bc == 0 and wc > 0:
            return True, 0
        if wc == 0 and bc == 0:
            return True, None
        return False, None

    def evaluate(self, player: int) -> float:
        """子力+轻推进, w(advance)=15 适度强化."""
        my_piece = "W" if player == 0 else "B"
        opp_piece = "B" if my_piece == "W" else "W"
        my_dir = 1 if player == 0 else -1

        material = 0.0
        advance = 0.0

        for idx, p in enumerate(self._pieces):
            if p == "":
                continue
            _, row = self._col_row(idx)
            if p == my_piece:
                material += 1.0
                if my_dir == 1:
                    advance += (row - 1) / 14.0
                else:
                    advance += (8 - row) / 14.0
            else:
                material -= 1.0
                if my_dir == 1:
                    advance -= (row - 1) / 14.0
                else:
                    advance -= (8 - row) / 14.0

        return material * 10.0 + advance * 15.0

    def move_key(self, action: str) -> int:
        """Move ordering: 吃子按目标行 > 推进按行 > 中心优先."""
        m = _MOVE_RE.match(action)
        if not m:
            return -1
        from_sq = self._coord(m.group(1))
        to_sq = self._coord(m.group(2))
        to_col, to_row = self._col_row(to_sq)

        my_piece = "W" if self._player_to_move == 0 else "B"
        opp_piece = "B" if my_piece == "W" else "W"
        my_dir = 1 if my_piece == "W" else -1

        priority = 0
        # 吃子: 最高优先级
        if self._pieces[to_sq] == opp_piece:
            priority += 10000
        # 推进距离 (步数)
        priority += (to_row - self._col_row(from_sq)[1]) * my_dir * 500
        # 中心偏好
        priority += (4 - abs(to_col - 3.5)) * 100
        # 直达底线 = 获胜
        if (my_piece == "W" and to_row == 8) or (my_piece == "B" and to_row == 1):
            priority += 100000
        return priority

    # ------------------------------------------------------------------
    # Hash / eq
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash(self._pieces)

    def __eq__(self, other) -> bool:
        return isinstance(other, BreakthroughState) and self._pieces == other._pieces

    def __repr__(self) -> str:
        return f"BT(P{self._player_to_move})"
