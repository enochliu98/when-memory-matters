"""
TicTacToe optimal opponent (perfect minimax with transposition table).

TextArena conventions (verified against textarena.envs.TicTacToe.env):
  - Player 0 plays as 'O' and moves FIRST
  - Player 1 plays as 'X'
  - Observation header always contains "You are Player N"
  - Observation is CUMULATIVE — every step appends action descriptions and
    a fresh "Current Board:" rendering. The latest board is the LAST board
    block in the obs; older boards remain visible above. We must parse the
    LAST board, never the first.

Search:
  - 9-cell state space (~5478 reachable positions). Full negamax with a
    transposition table over `tuple(cells)` keys terminates in milliseconds.
  - Terminal scores: ±1 for win/loss, 0 for draw, with a small depth bonus
    so faster wins / slower losses are preferred over equally-valued lines.
"""

import math
import re
from typing import List, Optional, Tuple

from .base import BaseOpponent


# Matches a single TTT row of the rendered board: " X | 1 | 2 ".
# Cells are either 'X', 'O', or a single digit (the cell-index for empty).
_TTT_ROW_RE = re.compile(r" ([XO\d]) \| ([XO\d]) \| ([XO\d]) ")
_PLAYER_ID_RE = re.compile(r"You are Player (\d+)")
_LEGAL_IDX_RE = re.compile(r"\d+")


def _opp(sym: str) -> str:
    return 'O' if sym == 'X' else 'X'


class TicTacToeMinimax(BaseOpponent):
    """Perfect-play TicTacToe opponent via full-depth negamax."""

    def __init__(self, max_depth: int = 9):
        super().__init__("ttt_minimax")
        # max_depth is kept for signature parity with the previous class.
        # TTT has at most 9 plies, so any value ≥ 9 is equivalent.
        self.max_depth = max_depth
        self._tt: dict = {}

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_player_id(observation: str) -> int:
        m = _PLAYER_ID_RE.search(observation)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _parse_last_board(observation: str) -> List[str]:
        """Extract the 9 cells from the LAST board block in `observation`.

        Returns a 9-element list with 'X' / 'O' for occupied cells and ''
        for empty cells. If no board can be found, returns an empty board.
        """
        rows = _TTT_ROW_RE.findall(observation)
        if len(rows) < 3:
            return [''] * 9
        # Each board block contributes 3 consecutive row tuples in order
        # (top→bottom). The last 3 tuples are the latest board.
        last_three = rows[-3:]
        cells: List[str] = []
        for row in last_three:
            for c in row:
                cells.append(c if c in ('X', 'O') else '')
        return cells

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        if not legal_actions:
            return ""
        cells = self._parse_last_board(observation)
        player_id = self._parse_player_id(observation)
        my_symbol = 'O' if player_id == 0 else 'X'

        best_value = -math.inf
        best_move = legal_actions[0]
        for move in legal_actions:
            m = _LEGAL_IDX_RE.search(move)
            if not m:
                continue
            idx = int(m.group())
            if idx < 0 or idx > 8 or cells[idx] != '':
                continue
            new_cells = list(cells)
            new_cells[idx] = my_symbol
            # After I move, it's opponent's turn — negamax recursion.
            value = -self._negamax(tuple(new_cells), _opp(my_symbol))
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def _negamax(self, cells: Tuple[str, ...], to_move: str) -> float:
        """Negamax value of `cells` from `to_move`'s perspective.

        `to_move` is the symbol that is about to play. The position has
        already had the previous player's move applied. If the previous
        player just won, we are at a terminal state from to_move's POV
        (to_move LOST).
        """
        if cells in self._tt:
            return self._tt[cells]

        winner = self._winner(cells)
        if winner is not None:
            # The previous mover won → to_move lost. Prefer LATER losses:
            # depth = filled-cell count; large depth → smaller penalty.
            depth = sum(1 for c in cells if c)
            value = -1.0 + depth * 0.01
            self._tt[cells] = value
            return value
        if all(cells):
            self._tt[cells] = 0.0
            return 0.0

        best = -math.inf
        for i, c in enumerate(cells):
            if c:
                continue
            new_cells = list(cells)
            new_cells[i] = to_move
            v = -self._negamax(tuple(new_cells), _opp(to_move))
            if v > best:
                best = v
        self._tt[cells] = best
        return best

    @staticmethod
    def _winner(cells) -> Optional[str]:
        """Return 'X' or 'O' if there is a winner, else None."""
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6),              # diagonals
        )
        for a, b, c in lines:
            v = cells[a]
            if v and v == cells[b] == cells[c]:
                return v
        return None
