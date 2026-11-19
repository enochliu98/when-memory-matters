"""
Deterministic rule-based reference opponent for TextArena KuhnPoker.

TextArena conventions (verified against textarena.envs.KuhnPoker.env):
  - Two players, 3 rounds per game, J<Q<K
  - Each player antes 1 chip per round, gets 1 card
  - Actions (lowercased inside brackets):
      [check] / [bet]   — opening (no bet on the table)
      [call]  / [fold]  — facing a bet
  - Observation is CUMULATIVE — only the latest "### Starting round N"
    block carries the live card. We re-parse for the live card each turn.
  - Player id appears in the obs header ("You are Player N in a 3 round ...").

Heuristic action table, not a mixed Nash-equilibrium policy:

    Card  | opening (no bet) | facing a bet
    ------|------------------|--------------
    J     | check            | fold
    Q     | bet              | call
    K     | bet              | call
"""

import re
from typing import List

from .base import BaseOpponent


_CARD_RE = re.compile(r"Your card is:\s*'([JQK])'")
_PLAYER_ID_RE = re.compile(r"You are Player (\d+)")


class KuhnPokerOptimal(BaseOpponent):
    """Plays the historical deterministic Kuhn Poker reference policy."""

    def __init__(self):
        super().__init__("kuhn_optimal")

    @staticmethod
    def _parse_card(observation: str) -> str:
        """Extract my card from the LATEST '### Starting round N' block.

        TextArena's obs is cumulative; each round appends a fresh
        "Your card is: 'X'" line. We must take the last occurrence,
        otherwise we lock onto round 1's card for the whole game.
        """
        matches = _CARD_RE.findall(observation)
        return matches[-1] if matches else ""

    @staticmethod
    def _parse_player_id(observation: str) -> int:
        m = _PLAYER_ID_RE.search(observation)
        return int(m.group(1)) if m else 0

    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        if not legal_actions:
            return ""
        card = self._parse_card(observation)
        # In TextArena's KuhnPoker, when opponent has bet the legal action
        # set is [call]/[fold]; when nobody has bet it's [check]/[bet].
        # Mid-round responses (after a call) revert to check/bet.
        legal_set = set(legal_actions)
        facing_bet = "[call]" in legal_set or "[fold]" in legal_set

        if card == "J":
            if "[fold]" in legal_set:
                return "[fold]"
            return "[check]"

        # Q and K: when facing a bet, call; when opening, bet.
        if facing_bet:
            if "[call]" in legal_set:
                return "[call]"
            # Defensive: should not happen.
            return "[check]"
        # Opening
        if "[bet]" in legal_set:
            return "[bet]"
        if "[check]" in legal_set:
            return "[check]"
        return legal_actions[0]
