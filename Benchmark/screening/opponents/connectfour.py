"""
ConnectFour minimax opponent (alpha-beta + heuristic evaluation).

TextArena conventions (verified against textarena.envs.ConnectFour.env):
  - Player 0 plays as 'X' and moves FIRST
  - Player 1 plays as 'O'
  - Action format: '[col N]' or '[N]' (the action regex accepts both;
    we always emit '[N]' which matches the legal_actions strings the
    NonSaturationEvaluator generates).
  - Observation is CUMULATIVE — every step appends the action description
    AND a fresh "Board state:\\n0 1 2 3 4 5 6\\n-------------\\n<6 rows>"
    block. We MUST parse the LAST block.
  - Board orientation: TextArena's renderer joins board rows in order, so
    row 0 = TOP of the board, row 5 = BOTTOM. Discs settle to the highest
    available row index — i.e. the bottom (row 5) fills first, then row 4,
    etc.

Search:
  - ConnectFour is solved (first player wins with perfect play, ~10^14
    states) but full search is infeasible. We do alpha-beta to a bounded
    depth (default 6) with center-first move ordering for stronger pruning,
    plus a positional + threat heuristic at the leaves.
  - Internal state: tuple of 6 strings of length 7 (immutable, hashable).
  - The depth bonus on terminal scores (`±10000 ∓ depth`) makes the agent
    prefer faster wins / slower losses among equally-valued lines.
"""

import math
import re
from typing import List, Tuple

from .base import BaseOpponent


_PLAYER_ID_RE = re.compile(r"You are Player (\d+)")
_LEGAL_IDX_RE = re.compile(r"\d+")
_BOARD_HEADER_RE = re.compile(r"^\s*0\s+1\s+2\s+3\s+4\s+5\s+6\s*$")
_BOARD_SEPARATOR_RE = re.compile(r"^-{5,}$")
_BOARD_ROW_RE = re.compile(r"^([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s+([.XO])\s*$")

NUM_ROWS = 6
NUM_COLS = 7
# Center-first column ordering — better alpha-beta pruning.
_COL_ORDER = (3, 2, 4, 1, 5, 0, 6)
# Positional weights, indexed by [row][col]. Row 0 = top. The classic
# ConnectFour heuristic favours center cells, slightly higher for cells
# in the middle rows where they participate in more potential lines.
_POSITION_WEIGHTS = (
    (3, 4, 5, 7, 5, 4, 3),
    (4, 6, 8, 10, 8, 6, 4),
    (5, 8, 11, 13, 11, 8, 5),
    (5, 8, 11, 13, 11, 8, 5),
    (4, 6, 8, 10, 8, 6, 4),
    (3, 4, 5, 7, 5, 4, 3),
)
TERMINAL_WIN = 100_000.0


def _opp(sym: str) -> str:
    return 'O' if sym == 'X' else 'X'


class ConnectFourMinimax(BaseOpponent):
    """Alpha-beta minimax opponent for ConnectFour."""

    def __init__(self, max_depth: int = 6):
        super().__init__("c4_minimax")
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_player_id(observation: str) -> int:
        m = _PLAYER_ID_RE.search(observation)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _parse_last_board(observation: str) -> Tuple[Tuple[str, ...], ...]:
        """Extract the LAST 6x7 board from `observation`.

        Returns a tuple of 6 tuples (row 0 = top). Empty cells are '.',
        occupied cells are 'X' or 'O'. If parsing fails, returns the
        empty board.
        """
        lines = observation.splitlines()
        # Scan backwards for the last "0 1 2 3 4 5 6" header line. The 6
        # rows immediately follow the dashed separator that comes right
        # after this header.
        for i in range(len(lines) - 1, -1, -1):
            if not _BOARD_HEADER_RE.match(lines[i]):
                continue
            # Expect a dashed separator on i+1, then 6 board rows.
            if i + 1 >= len(lines) or not _BOARD_SEPARATOR_RE.match(lines[i + 1]):
                continue
            rows: List[Tuple[str, ...]] = []
            for k in range(i + 2, min(i + 2 + NUM_ROWS, len(lines))):
                m = _BOARD_ROW_RE.match(lines[k])
                if not m:
                    break
                rows.append(tuple(m.groups()))
            if len(rows) == NUM_ROWS:
                return tuple(rows)
        return tuple(tuple('.' for _ in range(NUM_COLS)) for _ in range(NUM_ROWS))

    @staticmethod
    def _legal_columns(board: Tuple[Tuple[str, ...], ...]) -> List[int]:
        """Columns that still have at least one empty slot (top row '.')."""
        return [c for c in range(NUM_COLS) if board[0][c] == '.']

    @staticmethod
    def _drop(board: Tuple[Tuple[str, ...], ...], col: int, sym: str) -> Tuple[Tuple[str, ...], ...]:
        """Drop `sym` into `col`. Caller guarantees the column isn't full."""
        # Find the lowest empty row (largest row index with '.').
        for r in range(NUM_ROWS - 1, -1, -1):
            if board[r][col] == '.':
                new_row = list(board[r])
                new_row[col] = sym
                return board[:r] + (tuple(new_row),) + board[r + 1:]
        # Should never get here if caller checked legality.
        return board

    # ------------------------------------------------------------------
    # Win detection
    # ------------------------------------------------------------------
    @staticmethod
    def _last_drop_row(board: Tuple[Tuple[str, ...], ...], col: int) -> int:
        """Row index of the topmost piece in `col` (i.e. the most recent
        drop's landing row). Returns NUM_ROWS if the column is empty."""
        for r in range(NUM_ROWS):
            if board[r][col] != '.':
                return r
        return NUM_ROWS

    @staticmethod
    def _is_win_at(board: Tuple[Tuple[str, ...], ...], r: int, c: int) -> bool:
        """Did the piece at (r, c) just complete a 4-in-a-row?"""
        sym = board[r][c]
        if sym == '.':
            return False
        # Four directions: horizontal, vertical, two diagonals.
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            # Walk forward
            rr, cc = r + dr, c + dc
            while 0 <= rr < NUM_ROWS and 0 <= cc < NUM_COLS and board[rr][cc] == sym:
                count += 1
                rr += dr
                cc += dc
            # Walk backward
            rr, cc = r - dr, c - dc
            while 0 <= rr < NUM_ROWS and 0 <= cc < NUM_COLS and board[rr][cc] == sym:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 4:
                return True
        return False

    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, board: Tuple[Tuple[str, ...], ...], me: str) -> float:
        """Heuristic score from `me`'s perspective (positive = good for me).

        Sums positional weights and per-line "open 3s" (3 of mine + 1 empty
        in any 4-cell window). This is intentionally simple — at the cost
        of perfect endgame insight (which alpha-beta backs up anyway) we
        keep evaluation cheap so depth=6 stays interactive.
        """
        opp = _opp(me)
        score = 0.0

        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                v = board[r][c]
                if v == me:
                    score += _POSITION_WEIGHTS[r][c]
                elif v == opp:
                    score -= _POSITION_WEIGHTS[r][c]

        # Open-three threats: scan all 4-cell windows in each direction.
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rr, cc = r + 3 * dr, c + 3 * dc
                    if not (0 <= rr < NUM_ROWS and 0 <= cc < NUM_COLS):
                        continue
                    window = [board[r + i * dr][c + i * dc] for i in range(4)]
                    me_count = window.count(me)
                    opp_count = window.count(opp)
                    empty_count = window.count('.')
                    if opp_count == 0 and me_count == 3 and empty_count == 1:
                        score += 50
                    elif opp_count == 0 and me_count == 2 and empty_count == 2:
                        score += 5
                    elif me_count == 0 and opp_count == 3 and empty_count == 1:
                        score -= 60   # Block opponent threats slightly more
                                      # eagerly than we build our own.
                    elif me_count == 0 and opp_count == 2 and empty_count == 2:
                        score -= 5
        return score

    # ------------------------------------------------------------------
    # Alpha-beta search
    # ------------------------------------------------------------------
    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        if not legal_actions:
            return ""
        board = self._parse_last_board(observation)
        player_id = self._parse_player_id(observation)
        my_symbol = 'X' if player_id == 0 else 'O'

        # Build a quick lookup from col -> action string so we return the
        # exact legal_actions string (not a freshly constructed one).
        col_to_action = {}
        for action in legal_actions:
            m = _LEGAL_IDX_RE.search(action)
            if m:
                col_to_action[int(m.group())] = action

        legal_cols = [c for c in self._legal_columns(board) if c in col_to_action]
        if not legal_cols:
            return legal_actions[0]
        # Center-first ordering for the root.
        legal_cols.sort(key=lambda c: _COL_ORDER.index(c) if c in _COL_ORDER else 99)

        best_value = -math.inf
        best_col = legal_cols[0]
        alpha, beta = -math.inf, math.inf
        for col in legal_cols:
            child = self._drop(board, col, my_symbol)
            # Did I just win? Take it.
            r = self._last_drop_row(child, col)
            if self._is_win_at(child, r, col):
                return col_to_action[col]
            v = -self._negamax(child, _opp(my_symbol), my_symbol,
                                self.max_depth - 1, -beta, -alpha)
            if v > best_value:
                best_value = v
                best_col = col
            if v > alpha:
                alpha = v
        return col_to_action[best_col]

    def _negamax(
        self,
        board: Tuple[Tuple[str, ...], ...],
        to_move: str,
        me: str,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Negamax with alpha-beta, returning value from `to_move`'s POV.

        We do NOT memoize across moves — the transposition cache would need
        depth-aware bounds to be safe, and the simple form was found to be
        a net loss vs memory churn at depth 6. Skipping it keeps the code
        small and correct.
        """
        legal_cols = self._legal_columns(board)
        if not legal_cols:
            return 0.0
        if depth <= 0:
            return self._evaluate(board, to_move)

        legal_cols.sort(key=lambda c: _COL_ORDER.index(c) if c in _COL_ORDER else 99)

        best = -math.inf
        for col in legal_cols:
            child = self._drop(board, col, to_move)
            r = self._last_drop_row(child, col)
            if self._is_win_at(child, r, col):
                # to_move just won — score for to_move is +TERMINAL_WIN,
                # discounted by remaining depth so we prefer faster wins.
                v = TERMINAL_WIN - (self.max_depth - depth)
            else:
                v = -self._negamax(child, _opp(to_move), me, depth - 1,
                                    -beta, -alpha)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
