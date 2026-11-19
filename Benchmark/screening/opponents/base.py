"""
Base classes for programmatic opponents used in the GameScreener.

Only two opponents live here:
  - BaseOpponent:    the shared interface
  - RandomOpponent:  uniform random over legal moves

Game-specific optimal/minimax opponents (TicTacToe, ConnectFour, Nim) live
in their own files and inherit from BaseOpponent directly. They each parse
the TextArena observation themselves — no shared evaluate_board/apply_move
hooks, because the observation format and search representation differ
enough between games that abstracting them was net-negative complexity.
"""

import random
from typing import List


class BaseOpponent:
    """Base class for all opponents."""

    def __init__(self, name: str = "base"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        """Select an action given the current observation and legal moves."""
        raise NotImplementedError


class RandomOpponent(BaseOpponent):
    """Plays uniformly random legal moves."""

    def __init__(self):
        super().__init__("random")

    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        return random.choice(legal_actions) if legal_actions else ""
