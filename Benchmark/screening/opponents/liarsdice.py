"""
Liar's Dice strong heuristic opponent.

TextArena conventions (verified against textarena.envs.LiarsDice.env):
  - 2 players, each starts with 5 dice. Loser of a call loses one die.
  - Action format: '[Bid: <qty>, <face>]' (raise) or '[Call]' (challenge)
  - Observation is CUMULATIVE — every step appends a fresh
    "Your current Dice are: ..." line, the latest bid line, and the
    action history. We MUST parse the LATEST snapshot.
  - Player id appears in the obs header ("You are Player N in an 2-player
    Liar's Dice game.").

Strategy (no closed-form GTO for Liar's Dice; we use a strong, well-known
heuristic that reliably beats random play):

  1. Parse my dice and the current bid (qty, face) from the LAST
     snapshot in the observation.
  2. Compute my_match = number of my dice equal to the bid face.
  3. Estimate P(actual count of face >= bid qty), modelling opponent's
     dice as uniform. If that probability is below a call threshold,
     challenge with [Call].
  4. Otherwise, raise with a sensible bid: prefer to increase quantity
     (cheaper for our range), using either the bid face or the face we
     have the most of.

This is a "strong" opponent in the sense that it exploits the most
common LLM/random failure (over-bluffing); it is not necessarily GTO,
but it is a stable, deterministic reference for screening.
"""

import random
import re
from collections import Counter
from typing import List, Optional, Tuple

from .base import BaseOpponent


_MY_DICE_RE = re.compile(r"Your current Dice are:\s*([0-9 ,]+)")
_REMAINING_RE = re.compile(
    r"Remaining dice:\s*Player\s+(\d+):\s*(\d+);\s*Player\s+(\d+):\s*(\d+)"
)
_BID_LINE_RE = re.compile(r"bids\s+(\d+)\s+of\s+face\s+(\d+)\.")
_PLAYER_ID_RE = re.compile(r"You are Player (\d+)")


def _binom_p_at_least(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Cumulative from k..n."""
    if k > n:
        return 0.0
    if k <= 0:
        return 1.0
    # Recursive CDF is fine for small n (opponent's remaining dice ≤ 5).
    cdf = 0.0
    for i in range(0, k):
        cdf += _binom_pmf(i, n, p)
    return max(0.0, min(1.0, 1.0 - cdf))


def _binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p)."""
    if k < 0 or k > n:
        return 0.0
    # log-PMF for numerical stability
    from math import log
    from math import exp
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    try:
        log_pmf = (
            _binom_log_coef(n, k)
            + k * log(p)
            + (n - k) * log(1.0 - p)
        )
        return exp(log_pmf)
    except (ValueError, OverflowError):
        return 0.0


def _binom_log_coef(n: int, k: int) -> float:
    """log(C(n, k)) via lgamma for stability."""
    from math import lgamma
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


class LiarsDiceOptimal(BaseOpponent):
    """Strong heuristic opponent for Liar's Dice."""

    # Probability threshold below which we challenge a bid. 0.30 is
    # reasonable: it accepts bids that are "obviously true" (e.g. bid
    # of 1 of my dominant face) but challenges bids that would require
    # the opponent to be carrying many of the claimed face.
    CALL_THRESHOLD = 0.30

    def __init__(self, call_threshold: float = CALL_THRESHOLD):
        super().__init__("liarsdice_strong")
        self.call_threshold = call_threshold

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_my_dice(observation: str) -> List[int]:
        """Extract my dice from the LAST 'Your current Dice are: ...' line."""
        # Search from the end.
        idx = observation.rfind("Your current Dice are:")
        if idx < 0:
            return []
        tail = observation[idx:]
        m = _MY_DICE_RE.search(tail)
        if not m:
            return []
        try:
            return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        except ValueError:
            return []

    @staticmethod
    def _parse_remaining(observation: str) -> Tuple[int, int]:
        """Extract remaining dice counts for both players."""
        idx = observation.rfind("Remaining dice:")
        if idx < 0:
            return (0, 0)
        m = _REMAINING_RE.search(observation[idx:])
        if not m:
            return (0, 0)
        # The two (pid, count) pairs may be in any order. Sort by pid.
        a_pid, a_cnt = int(m.group(1)), int(m.group(2))
        b_pid, b_cnt = int(m.group(3)), int(m.group(4))
        ordered = sorted([(a_pid, a_cnt), (b_pid, b_cnt)], key=lambda x: x[0])
        return ordered[0][1], ordered[1][1]

    @staticmethod
    def _parse_player_id(observation: str) -> int:
        m = _PLAYER_ID_RE.search(observation)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _parse_current_bid(observation: str) -> Optional[Tuple[int, int]]:
        """Find the LAST bid line in the obs: '[GAME] Player X bids N of face F.'"""
        matches = _BID_LINE_RE.findall(observation)
        if not matches:
            return None
        try:
            qty, face = int(matches[-1][0]), int(matches[-1][1])
            return qty, face
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    def _estimate_truth_prob(self, my_match: int, opp_remaining: int,
                             bid_qty: int) -> float:
        """P(opponent contributes at least (bid_qty - my_match) of the face)
        assuming opponent's dice are uniform over 1..6.

        Returns P(actual count >= bid_qty) where actual = my_match + opp's count.
        """
        need = max(0, bid_qty - my_match)
        if opp_remaining <= 0:
            return 1.0 if my_match >= bid_qty else 0.0
        return _binom_p_at_least(need, opp_remaining, 1.0 / 6.0)

    def _opening_bid(self, my_dice: List[int], legal: List[str]) -> str:
        """No bid on the table yet — open with a small bid using my
        most common face (or face 1 if all distinct)."""
        if not my_dice:
            return "[Bid: 1, 1]" if "[Bid: 1, 1]" in legal else legal[0]
        most_common_face = Counter(my_dice).most_common(1)[0][0]
        # Try (1, face) first; if not legal, pick the cheapest legal opening.
        preferred = f"[Bid: 1, {most_common_face}]"
        if preferred in legal:
            return preferred
        for a in legal:
            if a.startswith("[Bid: 1,"):
                return a
        return legal[0]

    def _raise_bid(self, my_dice: List[int],
                   cur_bid: Tuple[int, int],
                   legal: List[str]) -> str:
        """Build a sensible raise. Prefer to bump quantity on the
        bid face, else switch to a face we have more of."""
        cur_q, cur_f = cur_bid
        # Try the cheapest legal raise: (cur_q+1, cur_f)
        preferred = f"[Bid: {cur_q + 1}, {cur_f}]"
        if preferred in legal:
            return preferred
        # Otherwise: face change. Find a face I have most of.
        if my_dice:
            target_face = Counter(my_dice).most_common(1)[0][0]
        else:
            target_face = 1
        # Try (cur_q, target_face) if it's strictly higher
        if (cur_q, target_face) > cur_bid:
            preferred = f"[Bid: {cur_q}, {target_face}]"
            if preferred in legal:
                return preferred
        # Fallback: scan legal actions in order, pick the first Bid.
        for a in legal:
            if a.startswith("[Bid:"):
                return a
        return legal[0]

    def select_action(self, observation: str, legal_actions: List[str]) -> str:
        if not legal_actions:
            return ""

        # [Call] may or may not be in the legal set (the env generally
        # accepts it whenever a bid is on the table).
        my_dice = self._parse_my_dice(observation)
        my_remaining, opp_remaining = self._parse_remaining(observation)
        cur_bid = self._parse_current_bid(observation)

        if cur_bid is None:
            return self._opening_bid(my_dice, legal_actions)

        cur_q, cur_f = cur_bid
        my_match = sum(1 for d in my_dice if d == cur_f)
        truth_prob = self._estimate_truth_prob(my_match, opp_remaining, cur_q)

        # Decide: call if the bid is unlikely to be true.
        if "[Call]" in legal_actions and truth_prob < self.call_threshold:
            return "[Call]"

        # Otherwise raise. If the truth probability is moderate, still
        # raise (caller of raise only loses if they claim a bid the next
        # player can challenge and find false). The raise policy is
        # intentionally conservative (lowest legal face/quantity bump).
        return self._raise_bid(my_dice, cur_bid, legal_actions)
