"""
Nim optimal opponent (XOR / Nim-sum strategy).

TextArena conventions (verified against textarena.envs.Nim.env):
  - Two players, no symbols. Player 0 moves first.
  - Action format: '[pile_index quantity]', e.g. '[0 3]'
  - Observation is CUMULATIVE — every step appends a fresh "Current Pile:"
    block. We MUST parse the LAST block; old blocks remain visible.
  - Initial state is configured at env-registration time (e.g. [3,4,5]
    for Nim-v0). Number of piles is fixed but unknown to the opponent;
    we infer it from the parsed pile lines.

Strategy:
  - Compute Nim-sum = XOR of all pile sizes.
  - If Nim-sum != 0, current position is winning. Choose pile i where
    piles[i] ^ nim_sum < piles[i] and remove piles[i] - target objects.
  - If Nim-sum == 0, current position is losing — any move is suboptimal.
    We fall back to a uniform random legal move (sufficient for screening;
    the LLM-as-Optimal use-case is win-rate measurement, not adversarial
    deception in lost positions).
"""

import random
import re
from typing import List

from .base import BaseOpponent


# Match a single pile line: "  pile 0: 3" (ignoring leading whitespace).
_PILE_LINE_RE = re.compile(r"^\s*pile\s+(\d+)\s*:\s*(\d+)\s*$", re.IGNORECASE)
# Marker that opens a Nim pile block in TextArena's renderer.
_PILE_BLOCK_MARKER = "Current Pile:"


class NimOptimal(BaseOpponent):
    """Plays optimally using the Nim-sum (XOR) strategy."""

    def __init__(self):
        super().__init__("nim_optimal")

    @staticmethod
    def _parse_last_piles(observation: str) -> List[int]:
        """Extract the pile sizes from the LAST 'Current Pile:' block.

        TextArena renders piles indexed 0..N-1, one per line. The block
        runs until a non-matching line. If multiple blocks are present
        (cumulative obs), the LAST block is the live state.
        """
        lines = observation.splitlines()
        # Walk from the end backwards to find the last marker line.
        marker_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if _PILE_BLOCK_MARKER in lines[i]:
                marker_idx = i
                break
        if marker_idx < 0:
            return []

        piles_by_index: dict = {}
        for i in range(marker_idx + 1, len(lines)):
            m = _PILE_LINE_RE.match(lines[i])
            if not m:
                # End of the block.
                break
            piles_by_index[int(m.group(1))] = int(m.group(2))

        if not piles_by_index:
            return []
        # Reconstruct by index order, padding any gaps with 0.
        max_idx = max(piles_by_index)
        return [piles_by_index.get(i, 0) for i in range(max_idx + 1)]

    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        if not legal_actions:
            return ""
        piles = self._parse_last_piles(observation)
        if not piles:
            return random.choice(legal_actions)

        nim_sum = 0
        for p in piles:
            nim_sum ^= p

        if nim_sum != 0:
            # Winning position — find the unique pile we can reduce so that
            # the remaining nim_sum is 0.
            for i, p in enumerate(piles):
                target = p ^ nim_sum
                if target < p:
                    remove = p - target
                    action = f"[{i} {remove}]"
                    if action in legal_actions:
                        return action
            # If we land here, the legal_actions list disagrees with our
            # pile reading (shouldn't happen). Fall through to random.

        # Losing position OR rare disagreement: pick a random legal move.
        return random.choice(legal_actions)
