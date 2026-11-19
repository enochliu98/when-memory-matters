"""
Othello (Reversi) 游戏状态 (8×8).

TextArena 约定:
  - Player 0 = Black (B, 先手), Player 1 = White (W)
  - Action: `[row, col]`, e.g. [2, 3], [3, 2]
  - 行 0 = top, 行 7 = bottom; 列 0 = left, 列 7 = right
  - 初始: B(4,3), B(3,4); W(3,3), W(4,4)
  - 无合法动作时自动 pass; 双方都 pass 则终局
"""

import re
from typing import List, Optional, Tuple

SIZE = 8
SQUARES = SIZE * SIZE

# 8 个方向: (dr, dc)
_DIRECTIONS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))

_PLAYER_RE = re.compile(r"You are Player (\d+)")
_ACTION_RE = re.compile(r"\[(\d+),\s*(\d+)\]")
_BOARD_LINE_RE = re.compile(r"^(\d+)\|([.|BW]+)\|")


class OthelloState:
    """Immutable Othello 状态。"""

    __slots__ = ("_pieces", "_player_to_move", "_pass_occurred")

    def __init__(self, pieces: Tuple[str, ...], player_to_move: int,
                 pass_occurred: bool = False):
        self._pieces = pieces
        self._player_to_move = player_to_move
        self._pass_occurred = pass_occurred

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------
    @classmethod
    def new_game(cls) -> "OthelloState":
        pieces = [""] * SQUARES
        pieces[27] = "W"  # (3,3)
        pieces[28] = "B"  # (3,4)
        pieces[35] = "B"  # (4,3)
        pieces[36] = "W"  # (4,4)
        return cls(tuple(pieces), player_to_move=0)  # Black 先手

    @classmethod
    def parse_from_obs(cls, observation: str, my_player_id: int) -> "OthelloState":
        """从 TextArena 累积 obs 解析棋盘。

        关键: 必须同时处理 `.` (空位), 否则上一块棋盘的棋子会残留。
        取最后 8 行作为当前棋盘块, 逐行逐格写入 (含清空)。
        """
        pieces = [""] * SQUARES
        matches = []
        for line in observation.splitlines():
            m = _BOARD_LINE_RE.match(line)
            if not m:
                continue
            row = int(m.group(1))
            if row < 0 or row >= SIZE:
                continue
            content = m.group(2).replace("|", "")
            matches.append((row, content))

        if not matches:
            return cls.new_game()

        # 只有最后 8 行是当前棋盘; 顺序覆盖确保空位被清空
        for row, content in matches[-8:]:
            for c, ch in enumerate(content[:SIZE]):
                pieces[row * SIZE + c] = ch if ch in ("B", "W") else ""

        bc = pieces.count("B")
        wc = pieces.count("W")
        # 总子数为奇 → 刚走完偶数步的人 = 该另一方走
        # 每步棋盘加一子: 总子数为偶 → Black 还没走 → 该 Black
        player_to_move = 0 if (bc + wc) % 2 == 0 else 1
        return cls(tuple(pieces), player_to_move)

    # ------------------------------------------------------------------
    # 搜索协议
    # ------------------------------------------------------------------
    @property
    def player_to_move(self) -> int:
        return self._player_to_move

    def _idx(self, r: int, c: int) -> int:
        return r * SIZE + c

    def _coord(self, idx: int) -> Tuple[int, int]:
        return idx // SIZE, idx % SIZE

    def _legal_moves(self, player: int) -> List[Tuple[int, int]]:
        my = "B" if player == 0 else "W"
        opp = "W" if player == 0 else "B"
        moves = []
        for r in range(SIZE):
            for c in range(SIZE):
                idx = self._idx(r, c)
                if self._pieces[idx] != "":
                    continue
        # 去掉调试打印
                for dr, dc in _DIRECTIONS:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < SIZE and 0 <= cc < SIZE):
                        continue
                    if self._pieces[self._idx(rr, cc)] != opp:
                        continue
                    # 继续沿方向走
                    rr += dr
                    cc += dc
                    while 0 <= rr < SIZE and 0 <= cc < SIZE:
                        pi = self._idx(rr, cc)
                        if self._pieces[pi] == my:
                            moves.append((r, c))
                            break
                        if self._pieces[pi] == "":
                            break
                        rr += dr
                        cc += dc
        return moves

    def get_legal_actions(self) -> List[str]:
        return [f"[{r}, {c}]" for r, c in self._legal_moves(self._player_to_move)]

    def apply_action(self, action: str) -> "OthelloState":
        m = _ACTION_RE.match(action)
        if not m:
            return self
        r, c = int(m.group(1)), int(m.group(2))
        my = "B" if self._player_to_move == 0 else "W"
        opp = "W" if self._player_to_move == 0 else "B"
        idx = self._idx(r, c)
        if not (0 <= idx < SQUARES) or self._pieces[idx] != "":
            return self

        new_pieces = list(self._pieces)
        new_pieces[idx] = my
        # 翻转
        for dr, dc in _DIRECTIONS:
            rr, cc = r + dr, c + dc
            to_flip = []
            while 0 <= rr < SIZE and 0 <= cc < SIZE:
                pi = self._idx(rr, cc)
                if self._pieces[pi] == opp:
                    to_flip.append((rr, cc))
                elif self._pieces[pi] == my:
                    for fr, fc in to_flip:
                        new_pieces[self._idx(fr, fc)] = my
                    break
                else:
                    break
                rr += dr
                cc += dc

        next_player = 1 - self._player_to_move
        # 如果对方无合法动作, env 自动 pass 给当前玩家继续走
        if not self._has_legal_move(next_player):
            if self._pass_occurred:
                return OthelloState(tuple(new_pieces), next_player, pass_occurred=True)
            # 对方 pass → 当前玩家继续走, 标记 pass_occurred
            return OthelloState(tuple(new_pieces), self._player_to_move, pass_occurred=True)
        return OthelloState(tuple(new_pieces), next_player, pass_occurred=False)

    def _has_legal_move(self, player: int) -> bool:
        """快速检查 player 是否有合法动作 (找到即返回, 不枚举全部)."""
        my = "B" if player == 0 else "W"
        opp = "W" if player == 0 else "B"
        for r in range(SIZE):
            for c in range(SIZE):
                if self._pieces[self._idx(r, c)] != "":
                    continue
                for dr, dc in _DIRECTIONS:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < SIZE and 0 <= cc < SIZE):
                        continue
                    if self._pieces[self._idx(rr, cc)] != opp:
                        continue
                    rr += dr
                    cc += dc
                    while 0 <= rr < SIZE and 0 <= cc < SIZE:
                        pi = self._idx(rr, cc)
                        if self._pieces[pi] == my:
                            return True
                        if self._pieces[pi] == "":
                            break
                        rr += dr
                        cc += dc
        return False

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        # 双 pass 或满盘 → 终局
        if self._pass_occurred and not self._has_legal_move(self._player_to_move):
            bc = self._pieces.count("B")
            wc = self._pieces.count("W")
            if bc > wc: return True, 0
            if wc > bc: return True, 1
            return True, None
        if all(self._pieces):
            bc = self._pieces.count("B")
            wc = self._pieces.count("W")
            if bc > wc: return True, 0
            if wc > bc: return True, 1
            return True, None
        return False, None

    def evaluate(self, player: int) -> float:
        """评估: frontier 主导早盘, 角落主导中盘, 子力仅尾盘起作用."""
        my = "B" if player == 0 else "W"
        opp = "W" if player == 0 else "B"

        bc = self._pieces.count(my)
        wc = self._pieces.count(opp)
        total = bc + wc
        disc = (bc - wc) / max(total, 1) * 10.0

        # 角落: 最重要
        corners = {(0,0): 30, (0,7): 30, (7,0): 30, (7,7): 30}
        bad = {(0,1): 8, (1,0): 8, (1,1): 8, (0,6): 8, (1,6): 8, (1,7): 8,
               (6,0): 8, (6,1): 8, (7,1): 8, (6,6): 8, (6,7): 8, (7,6): 8}
        c_sq = {(0,2): 3, (0,5): 3, (2,0): 3, (2,7): 3,
                (5,0): 3, (5,7): 3, (7,2): 3, (7,5): 3}
        pos = 0.0
        for (r, c), wt in corners.items():
            i = self._idx(r, c)
            if self._pieces[i] == my: pos += wt
            elif self._pieces[i] == opp: pos -= wt
        for (r, c), wt in bad.items():
            i = self._idx(r, c)
            if self._pieces[i] == my: pos -= wt
            elif self._pieces[i] == opp: pos += wt
        for (r, c), wt in c_sq.items():
            i = self._idx(r, c)
            if self._pieces[i] == my: pos -= wt
            elif self._pieces[i] == opp: pos += wt

        # Frontier(暴露棋子): 早盘中盘主信号. 高权重.
        my_front = 0; opp_front = 0
        for idx, p in enumerate(self._pieces):
            if p == "": continue
            r, c = idx // 8, idx % 8
            for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < 8 and 0 <= nc < 8 and self._pieces[nr*8+nc] == "":
                    if p == my: my_front += 1
                    else: opp_front += 1
                    break
        frontier = (opp_front - my_front) * 50.0

        return disc + pos + frontier

    def move_key(self, action: str) -> int:
        """Move ordering: 角落优先, 次优, 坏位置最后."""
        m = _ACTION_RE.match(action)
        if not m:
            return 0
        r, c = int(m.group(1)), int(m.group(2))
        corners = {(0,0): 100, (0,7): 100, (7,0): 100, (7,7): 100}
        edges = {(0,2): 80, (0,3): 80, (0,4): 80, (0,5): 80,
                 (7,2): 80, (7,3): 80, (7,4): 80, (7,5): 80}
        bad = {(0,1): -10, (1,0): -10, (1,1): -10,
               (0,6): -10, (1,6): -10, (1,7): -10,
               (6,0): -10, (6,1): -10, (7,1): -10,
               (6,6): -10, (6,7): -10, (7,6): -10}
        key = (r, c)
        if key in corners: return corners[key]
        if key in edges:   return edges[key]
        if key in bad:     return bad[key]
        return 0

    # ------------------------------------------------------------------
    # Hash / eq
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash((self._pieces, self._player_to_move, self._pass_occurred))

    def __eq__(self, other) -> bool:
        return (isinstance(other, OthelloState)
                and self._pieces == other._pieces
                and self._player_to_move == other._player_to_move)

    def __repr__(self) -> str:
        return f"Oth(P{self._player_to_move})"
