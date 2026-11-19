"""
TextArena Environment Wrapper

Provides an interface for running games in TextArena with LLM agents.
"""

import re
import textarena as ta
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Per-game action / observation helpers
# ---------------------------------------------------------------------------
# Each env registers:
#   - hint:           the LLM-facing "Output Format" hint
#   - move_pattern:   regex used by parse_action to extract "[N]" tokens
#   - move_range:     (lo, hi) inclusive legal-move range, used to reject
#                     tokens that look like moves but aren't in this game
#   - tail_extractor: function(observation: str) -> str, returns the
#                     "live" portion of the cumulative observation (the
#                     current board state + legal moves info). Default
#                     implementation looks for "[GAME] Current Board:" or
#                     "Current Board:".
#   - action_extractor: function(observation: str) -> List[str], returns
#                     the *currently legal* action strings (e.g. ['[1]',
#                     '[2]', ...]). Default implementation parses the
#                     "Available Moves:" line in the tail; games that do
#                     not emit one (e.g. ConnectFour) must provide a
#                     board-state-derived extractor.
# ---------------------------------------------------------------------------
def _default_tail_extractor(observation: str) -> str:
    """Locate the latest board snapshot in a TextArena observation."""
    for anchor in ("[GAME] Current Board:", "Current Board:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    return observation


def _default_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Default action extractor: find "Available Moves:" in the tail and
    parse every "[N]" token, filtered by env's move_range.
    """
    fmt = GAME_SPECS[env_id]
    lo, hi = fmt["move_range"]
    tail = fmt["tail_extractor"](observation)
    av_pos = tail.rfind("Available Moves:")
    moves_section = tail[av_pos:] if av_pos >= 0 else tail
    out: List[str] = []
    seen = set()
    for n in re.findall(fmt["move_pattern"], moves_section):
        if not n.isdigit():
            continue
        v = int(n)
        if v < lo or v > hi:
            continue
        candidate = f"[{n}]"
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Per-game helpers for the 4 new games (Nim, UltimateTicTacToe, Othello,
# Breakthrough). All four publish legal actions in the observation body
# (not in a dedicated "Available Moves:" line) so the default extractor
# does not apply.
# ---------------------------------------------------------------------------

def _extract_legal_from_text(text: str, pattern: str,
                             rebuild=None) -> List[str]:
    """Extract every regex match from `text` as a deduped list of
    full action strings. Order preserved. Used by the per-game
    extractors below to find the "Valid moves: '...'" list.

    `rebuild(matched) -> str` optionally turns a regex match (str or
    tuple of groups) into the final action token. When omitted, the
    default rebuilds single-group matches as `[<g>]` and multi-group
    matches as `[<g0> <g1>...]` (space-joined) — which matches Nim /
    UTTT but NOT Othello (comma) or Breakthrough (no separator).
    Callers whose env uses a different separator should pass `rebuild`.
    """
    out: List[str] = []
    seen = set()

    def _default_rebuild(m):
        if isinstance(m, tuple):
            return "[" + " ".join(m).strip() + "]"
        if m.startswith("["):
            return m
        return f"[{m}]"

    rb = rebuild or _default_rebuild
    for m in re.findall(pattern, text):
        s = rb(m)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _nim_tail_extractor(observation: str) -> str:
    """Nim: use the 'Current Pile:' section to the end of obs."""
    for anchor in ("Current Pile:", "[GAME] Current Pile:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    return observation


def _nim_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Nim actions look like '[<pile> <count>]' e.g. '[0 3]'. TextArena
    doesn't publish an explicit "Available Moves" line — players can
    remove from any non-empty pile. We don't try to enumerate legal
    moves from the pile counts (would require parsing pile values);
    the bench will instead rely on `parse_action` matching the
    LLM's output directly against the env's accepted format.
    """
    # Strategy: extract every '[N M]' token from the obs (examples
    # and pile contents alike), then dedupe. Move_range check skipped
    # because pile indices and quantities are game-state dependent.
    return _extract_legal_from_text(observation, r"\[(\d+)\s+(\d+)\]")


def _uttt_tail_extractor(observation: str) -> str:
    """UltimateTicTacToe: use the 'Current board:' (lowercase b) section."""
    for anchor in ("Current board:", "[GAME] Current board:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    return observation


def _uttt_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Derive the *currently legal* UTTT moves from the observation.

    Ultimate Tic-Tac-Toe actions look like '[<macro> <micro>]' e.g.
    '[7 8]'. The key rule is the POINTER MECHANIC: the opponent's
    last micro coordinate determines which macro board you MUST
    play in. If that board is already won/full, you get a FREE
    MOVE (any empty cell in any open board).

    The observation explicitly states "Player N must now play in
    micro board M" — we use that to determine the active macro
    board. We then parse the 9×9 board display to find which cells
    are still empty (still showing `[M,m]` with comma) versus taken
    (showing `X`/`O`). The legal set lets `parse_action` reject
    LLM outputs that try to play in the wrong board — the most
    common mistake that ends the game early.

    Falls back to the generic token extractor if the board cannot
    be parsed (e.g. malformed observation).
    """
    # 1) Determine the active macro board.
    must_play = re.findall(
        r"must now play in micro board (\d+)",
        observation, re.IGNORECASE,
    )
    if must_play:
        active_macro = int(must_play[-1])
    else:
        # First turn of the game: no directive yet, free move.
        active_macro = None  # sentinel: "free move"

    # 2) Parse the latest board display section to find empty cells.
    #    Board cells show as [M,m] with COMMA (e.g. '[4,3]').
    #    Taken cells are replaced by '  X   ' or '  O   '.
    #    We look only in the tail of the observation (after the
    #    LAST "Current board:" heading) to avoid stale cells from
    #    earlier historical board snapshots.
    tail = observation
    for anchor in ("[GAME] Current board:", "Current board:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            tail = observation[idx:]
            break

    # Collect all empty cells: (macro, micro) pairs from `[M,m]` tokens.
    empty_cells: List[Tuple[int, int]] = []
    seen: set = set()
    for m in re.finditer(r"\[(\d+),(\d+)\]", tail):
        macro_str, micro_str = m.group(1), m.group(2)
        key = (macro_str, micro_str)
        if key in seen:
            continue
        seen.add(key)
        macro, micro = int(macro_str), int(micro_str)
        if 0 <= macro <= 8 and 0 <= micro <= 8:
            empty_cells.append((macro, micro))

    if not empty_cells:
        # Board parsing failed; fall back to generic token extractor.
        return _extract_legal_from_text(observation, r"\[(\d+)\s+(\d+)\]")

    # 3) Build the per-macro-board open-cell index.
    board_open_cells: Dict[int, List[int]] = {}
    for macro, micro in empty_cells:
        board_open_cells.setdefault(macro, []).append(micro)

    open_boards = sorted(board_open_cells.keys())

    # 4) Generate legal actions.
    legal: List[str] = []
    visited: set = set()
    if active_macro is None or active_macro not in board_open_cells:
        # Free move: any empty cell in any open board.
        for macro in open_boards:
            for micro in board_open_cells[macro]:
                cand = f"[{macro} {micro}]"
                if cand not in visited:
                    visited.add(cand)
                    legal.append(cand)
    else:
        # Constrained move: only the active board.
        for micro in board_open_cells[active_macro]:
            cand = f"[{active_macro} {micro}]"
            if cand not in visited:
                visited.add(cand)
                legal.append(cand)
    return legal


def _othello_tail_extractor(observation: str) -> str:
    """Othello: use the '[GAME] Game Board:' section."""
    for anchor in ("[GAME] Game Board:", "Game Board:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    return observation


def _othello_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Othello actions look like '[row, col]' with a comma, e.g. '[2, 3]'.
    Extract from explicit "Valid moves:" lines plus any '[N, M]'
    token elsewhere in the obs. The rebuild uses ', ' so the token
    matches the env's expected shape (the default space-join would
    produce '[2 3]', which the env rejects).
    """
    return _extract_legal_from_text(
        observation, r"\[(\d+),\s*(\d+)\]",
        rebuild=lambda m: f"[{m[0]}, {m[1]}]",
    )


def _breakthrough_tail_extractor(observation: str) -> str:
    """Breakthrough: the board layout ends with a header row 'a b c d
    ... h'; everything from the first '[GAME]' after the rules text is
    the live board. Use a generic fallback that takes the last
    1200 chars (board is ~1000 chars)."""
    # The rules text ends with "Use UCI-like notation" or similar;
    # find the last [GAME] anchor (board state).
    for anchor in ("[GAME]", "Current Board:", "Board state:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    return observation[-1200:] if len(observation) > 1200 else observation


def _breakthrough_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Derive the *currently legal* Breakthrough moves from the board
    state in the observation.

    Breakthrough rules (2-player):
      - White moves UP (row 1 -> 8), Black moves DOWN (row 8 -> 1).
      - A piece may step one square straight-forward into an EMPTY
        square.
      - A piece may step one square diagonally-forward ONLY if an
        opponent piece occupies that square (a capture).
      - You may never move an opponent's piece, and you may never
        move straight-forward into an occupied square.

    The observation's last board block carries the live state and a
    `[White]` / `[Black]` tag identifying which side is to move. We
    parse that board, find our own pieces, and enumerate every legal
    (from, to) pair. Returning this legal set lets `parse_action`
    reject LLM outputs that try to move the wrong side (the most
    common failure mode that ends the game early with
    "That move does not follow the Breakthrough rules").

    Falls back to the generic token extractor if the board cannot be
    parsed (e.g. a malformed observation), so callers never crash.
    """
    # Find the side to move. The observation ends with the live
    # board and then a `[White]`/`[Black]` tag (or, for the very
    # first turn, an inline "You are playing White/Black" line).
    side = None
    m_side = re.search(r"You are playing (White|Black)", observation)
    if m_side:
        side = m_side.group(1)
    if side is None:
        # Last [White]/[Black] tag in the obs.
        tags = re.findall(r"\[(White|Black)\]", observation)
        if tags:
            side = tags[-1]
    if side is None:
        return _extract_legal_from_text(observation, r"\[([a-h]\d)([a-h]\d)\]")

    # Parse the LAST board block (the live one). Each data row looks
    # like: ` 8 | B B B B B B B B`. Column letters are a..h.
    board_rows: Dict[int, List[str]] = {}
    for m in re.finditer(
        r"^\s*(\d)\s*\|\s*((?:[WB.]\s*){8})\s*$",
        observation, re.MULTILINE,
    ):
        r = int(m.group(1))
        board_rows[r] = m.group(2).split()
    if not board_rows:
        return _extract_legal_from_text(observation, r"\[([a-h]\d)([a-h]\d)\]")

    cols = "abcdefgh"
    my_piece = "B" if side == "Black" else "W"
    opp_piece = "W" if side == "Black" else "B"
    # White moves up: row r -> r+1. Black moves down: row r -> r-1.
    fwd = 1 if side == "White" else -1

    def _at(col_idx: int, row: int) -> Optional[str]:
        cells = board_rows.get(row)
        if cells is None or not (0 <= col_idx < len(cells)):
            return None
        return cells[col_idx]

    legal: List[str] = []
    seen: set = set()
    for row, cells in board_rows.items():
        for ci, cell in enumerate(cells):
            if cell != my_piece:
                continue
            from_sq = f"{cols[ci]}{row}"
            to_row = row + fwd
            # Straight forward (must be empty).
            straight = _at(ci, to_row)
            if straight == ".":
                cand = f"[{from_sq}{cols[ci]}{to_row}]"
                if cand not in seen:
                    seen.add(cand)
                    legal.append(cand)
            # Diagonal forward-left / forward-right (must capture).
            for dci in (-1, 1):
                diag = _at(ci + dci, to_row)
                if diag == opp_piece:
                    cand = f"[{from_sq}{cols[ci + dci]}{to_row}]"
                    if cand not in seen:
                        seen.add(cand)
                        legal.append(cand)
    return legal


def _kuhnpoker_tail_extractor(observation: str) -> str:
    """
    Kuhn Poker's "live" state is per-round: the latest
    '### Starting round N' line carries the player's card and the
    available actions for this turn. Everything before is rule text
    and prior-round history.
    """
    for anchor in ("### Starting round", "[GAME] ### Starting round"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    # Fall back to the last [GAME] block.
    idx = observation.rfind("[GAME]")
    return observation[idx:] if idx >= 0 else observation


def _liarsdice_tail_extractor(observation: str) -> str:
    """
    Liar's Dice live state: the most recent "[GAME]" block carries the
    current "Your current Dice are: ..." and the latest bid/call lines.
    Use the LAST "[GAME]" anchor so the observation tail reflects the
    live round (not the static rules text at the top).
    """
    idx = observation.rfind("[GAME]")
    if idx >= 0:
        return observation[idx:]
    return observation


def _liarsdice_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Liar's Dice has two action shapes:

      - "[Bid: <qty>, <face>]"  -- raise over the current bid
        (strictly greater in quantity OR face value)
      - "[Call]"                -- challenge the current bid

    Strategy:
      1. Always advertise "[Call]" (always legal as long as a bid
         exists; first-mover with no prior bid can also call -- a
         no-op challenge -- but we still include it because the
         rules say "you may ... Call the last bid").
      2. Extract the current bid (quantity, face) from the latest
         "[GAME] Player X bids N of face F." line if present; else
         treat as "no bid yet" and advertise a small set of legal
         opening bids (quantity=1..3, face=1..6).
      3. Enumerate legal raises: any (q', f') with q' > q OR
         (q' == q AND f' > f). Cap face at 6.

    Note: we deliberately do NOT require every (q', f') to be
    reachable by a *single-step* monotonic raise over (q, f) per
    the env's "quantity OR face" rule. The env itself accepts a
    move like "[Bid: 1, 6]" over "[Bid: 5, 1]" (face jumps to 6)
    because face is now strictly greater. We just enumerate the
    union: every (q', f') with q' > q OR f' > f.
    """
    out: List[str] = []
    seen = set()

    # Always allow Call (the env treats "no prior bid" as a legal
    # no-op challenge but in practice the first move is always a
    # bid; including [Call] here is harmless).
    for tok in ("[Call]",):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)

    # Extract current bid from the latest "[GAME] Player X bids N of face F." line.
    cur_q = 0
    cur_f = 0
    m = re.findall(
        r"\[GAME\]\s+Player\s+\d+\s+bids\s+(\d+)\s+of\s+face\s+(\d+)\.",
        observation,
    )
    if m:
        try:
            cur_q = int(m[-1][0])
            cur_f = int(m[-1][1])
        except (ValueError, IndexError):
            cur_q = cur_f = 0

    # Enumerate legal raises.
    # q' can be at most (cur_q + 5): no hard upper bound from the
    # rules, but with 10 total dice (5 per player) the quantity
    # never exceeds 10 in practice. We cap at 12 to be safe.
    max_q = max(cur_q + 5, 6)
    for q in range(1, max_q + 1):
        for f in range(1, 7):
            if q > cur_q or f > cur_f:
                cand = f"[Bid: {q}, {f}]"
                if cand not in seen:
                    seen.add(cand)
                    out.append(cand)
    return out


def _kuhnpoker_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    Kuhn Poker actions are textual tokens: '[check]', '[bet]',
    '[call]', '[fold]'. TextArena publishes them on a line like
        "Your available actions are: '[check]', '[bet]'"
    We extract every such token (deduped, order-preserved) from
    the whole observation so the parser can match the LLM's reply.
    """
    out: List[str] = []
    seen = set()
    for tok in re.findall(r"\[(check|bet|call|fold)\]", observation):
        full = f"[{tok}]"
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def _connectfour_tail_extractor(observation: str) -> str:
    """
    ConnectFour uses "[GAME] Board state:" (not "Current Board:").
    Returns everything from the last such anchor to the end of obs.
    """
    for anchor in ("[GAME] Board state:", "Board state:"):
        idx = observation.rfind(anchor)
        if idx >= 0:
            return observation[idx:]
    return observation


def _connectfour_action_extractor(observation: str, env_id: str) -> List[str]:
    """
    ConnectFour does NOT advertise "Available Moves:" — we must derive
    legal actions from the board state. A column is legal iff its top
    cell (the first non-header row) is still '.' (i.e. not yet filled).

    The board looks like:
        0 1 2 3 4 5 6
        -------------
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . X O . . .
    Columns are 7 wide. The "top row" (first row of cells, after the
    header) tells us which columns are still playable.
    """
    fmt = GAME_SPECS[env_id]
    lo, hi = fmt["move_range"]
    tail = fmt["tail_extractor"](observation)
    if not tail:
        return []

    # Find the header line "0 1 2 3 4 5 6" to locate column indices.
    header_match = re.search(
        r"^([0-9](?:\s+[0-9])+)\s*$",
        tail,
        re.MULTILINE,
    )
    if not header_match:
        # No recognizable board — fall back to the full move range.
        return [f"[{i}]" for i in range(lo, hi + 1)]

    # Header tells us column count and order. Extract just the digits.
    cols = header_match.group(1).split()
    n_cols = len(cols)

    # First row of cells (the one directly under the "----" line).
    cell_lines: List[str] = []
    after_header = tail[header_match.end():]
    for line in after_header.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) <= set("- "):
            # Separator line.
            continue
        cell_lines.append(stripped)
        if len(cell_lines) == 1:
            break
    if not cell_lines:
        return [f"[{i}]" for i in range(lo, min(lo + n_cols, hi + 1))]

    top_row = cell_lines[0]
    # The row is space-separated single chars (e.g. ". . . X . . .").
    cells = top_row.split()
    if len(cells) != n_cols:
        return [f"[{i}]" for i in range(lo, min(lo + n_cols, hi + 1))]

    # Columns whose top cell is '.' are playable.
    out: List[str] = []
    for i, cell in enumerate(cells):
        if cell == "." and lo <= i <= hi:
            out.append(f"[{i}]")
    return out


# Per-env specs. New games plug in here with their own extractor.
GAME_SPECS: Dict[str, Dict] = {
    "TicTacToe-v0": {
        "hint": "Reply with exactly one move in the form '[N]' where N is a cell number 0-8 (e.g. '[4]').",
        "move_pattern": r"\[(\d+)\]",
        "move_range": (0, 8),
        "tail_extractor": _default_tail_extractor,
        "action_extractor": _default_action_extractor,
    },
    "ConnectFour-v0": {
        "hint": "Reply with exactly one move in the form '[N]' where N is a column number 0-6 (e.g. '[3]').",
        "move_pattern": r"\[(\d+)\]",
        "move_range": (0, 6),
        "tail_extractor": _connectfour_tail_extractor,
        "action_extractor": _connectfour_action_extractor,
    },
    "2048-v0": {
        "hint": "Reply with exactly one move in the form '[N]' where N is one of: 0=up, 1=down, 2=left, 3=right (e.g. '[0]').",
        "move_pattern": r"\[(\d+)\]",
        "move_range": (0, 3),
        "tail_extractor": _default_tail_extractor,
        "action_extractor": _default_action_extractor,
    },
    "Nim-v0": {
        "hint": "Reply with exactly one move in the form '[<pile> <count>]' e.g. '[0 3]' to remove 3 objects from pile 0.",
        "move_pattern": r"\[(\d+)\s+(\d+)\]",
        "move_range": (0, 999),
        "tail_extractor": _nim_tail_extractor,
        "action_extractor": _nim_action_extractor,
    },
    "UltimateTicTacToe-v0": {
        "hint": (
            "Reply with exactly one move as '[<macro> <micro>]' e.g. "
            "'[7 8]' (macro=which mini-board, micro=which square inside it). "
            "CRITICAL: the `macro` number is NOT your choice — it is "
            "determined by the opponent's last `micro`. Look in the "
            "observation for 'must now play in micro board M' and use "
            "that M as your macro. Only if that board is already won or "
            "full may you play any available cell (a 'free move')."
        ),
        "move_pattern": r"\[(\d+)\s+(\d+)\]",
        "move_range": (0, 8),
        "tail_extractor": _uttt_tail_extractor,
        "action_extractor": _uttt_action_extractor,
    },
    "Othello-v0": {
        "hint": "Reply with exactly one move in the form '[row, col]' e.g. '[2, 3]' (comma-separated).",
        "move_pattern": r"\[(\d+),\s*(\d+)\]",
        "move_range": (0, 7),
        "tail_extractor": _othello_tail_extractor,
        "action_extractor": _othello_action_extractor,
    },
    "Breakthrough-v0": {
        "hint": (
            "Reply with exactly one move in UCI-like notation "
            "'[<from><to>]' e.g. '[a2a3]'. The 'from' square MUST be "
            "occupied by YOUR OWN piece (check the observation to see "
            "which side you are and which rows your pieces are on). "
            "White moves UP (row 1->8); Black moves DOWN (row 8->1). "
            "A straight-forward move into an empty square is always "
            "legal; a diagonal move is only legal if it captures an "
            "opponent piece. Do not move an opponent's piece."
        ),
        "move_pattern": r"\[([a-h]\d)([a-h]\d)\]",
        "move_range": (0, 7),
        "tail_extractor": _breakthrough_tail_extractor,
        "action_extractor": _breakthrough_action_extractor,
    },
    "KuhnPoker-v0": {
        "hint": "Reply with exactly one action token: '[check]', '[bet]', '[call]', or '[fold]'.",
        "move_pattern": r"\[(check|bet|call|fold)\]",
        "move_range": (0, 0),
        "tail_extractor": _kuhnpoker_tail_extractor,
        "action_extractor": _kuhnpoker_action_extractor,
    },
    "LiarsDice-v0": {
        "hint": (
            "Reply with EXACTLY one action, either '[Bid: <qty>, <face>]' "
            "(e.g. '[Bid: 3, 4]' = 3 dice showing face 4) or '[Call]' to "
            "challenge the previous bid. The new bid must be strictly greater "
            "in quantity OR face than the previous bid (or any open bid if "
            "you're first to act this round). '1' is wild (matches any face)."
        ),
        # Match either [Bid: <qty>, <face>] OR the bare [Call] token.
        # We make the outer brackets non-grouping so the whole match
        # string is preserved (with surrounding [..]) when there's a
        # single group, and `re.findall` returns the bare groups only
        # when there are multiple groups (Nim / UTTT / Othello).
        # For Liar's Dice both alternatives produce a single token.
        "move_pattern": r"(?:\[Bid:\s*(\d+)\s*,\s*(\d+)\]|\[Call\])",
        "move_range": (0, 0),
        "tail_extractor": _liarsdice_tail_extractor,
        "action_extractor": _liarsdice_action_extractor,
    },
}


DEFAULT_GAME_SPEC: Dict = {
    "hint": "Reply with a single valid action in the format specified by the observation (typically '[N]' for board games, or a short natural-language message for conversational games).",
    "move_pattern": r"\[(\d+)\]",
    "move_range": (0, 99),
    "tail_extractor": _default_tail_extractor,
    "action_extractor": _default_action_extractor,
}


def get_action_format(env_id: str) -> Dict:
    """Return the per-env spec catalog for a given TextArena env."""
    return GAME_SPECS.get(env_id, DEFAULT_GAME_SPEC)


def extract_actions_from_observation(observation: str, env_id: str) -> List[str]:
    """
    Extract the set of *currently legal* action strings from an observation.

    The extraction strategy is registered per env in GAME_SPECS so each
    game can use its own logic (e.g. ConnectFour derives legal moves
    from the board state since the env does not emit "Available Moves:").
    """
    spec = get_action_format(env_id)
    extractor = spec["action_extractor"]
    return extractor(observation, env_id)


def parse_action(
    llm_output: str,
    env_id: str,
    observation: Optional[str] = None,
) -> str:
    """
    Parse the LLM's free-form response into a valid TextArena action string.

    Strategy (works for both single-token `[N]` actions and multi-token
    ones like `[N M]`, `[row, col]`, `[a1a2]`):

    1. Look for "Action: <token>" where <token> matches the env's
       `move_pattern` (handles `[N]`, `[N M]`, `[r, c]`, `[a1a2]`).
    2. Else, look for any full match of `move_pattern` in the output
       that is in the legal action set (when observation is provided).
    3. Else, look for the first full match of `move_pattern` in the
       output (last resort).
    4. Else, treat the whole stripped output as the action.

    Args:
        llm_output: Raw text from the LLM.
        env_id: TextArena env id (used to look up legal actions).
        observation: Current observation (used to filter to legal actions).

    Returns:
        A string that can be passed to env.step().
    """
    if llm_output is None:
        return ""

    text = llm_output.strip()
    fmt = get_action_format(env_id)
    pattern = fmt["move_pattern"]

    def _candidate(matched) -> str:
        """Turn a regex match (str or tuple-of-strs) into an action string.

        For single-group patterns like `\\[(\\d+)\\]`, the match string
        already includes the surrounding brackets (e.g. `'[4]'`), so we
        return it as-is. For multi-group patterns like `\\[(\\d+)\\s+(\\d+)\\]`,
        `re.findall` returns tuples of groups; we join them with a
        single space and re-wrap in brackets.

        For text-token games like Kuhn Poker whose pattern is
        `\\[(check|bet|call|fold)\\]`, `re.findall` returns the bare
        token (e.g. `'check'`) WITHOUT brackets — so we wrap it here.
        We detect this case by checking whether `matched` already
        starts with '['; if not, we add the brackets.

        Special case: Liar's Dice's pattern is `(?:\\[Bid: (\\d+), (\\d+)\\]|\\[Call\\])`
        with two inner groups. `re.findall` returns the inner groups
        for any match (filling unmatched groups with ''), so a `[Call]`
        match yields `('', '')`. We reconstruct by inserting the
        literal prefix when both groups are empty, otherwise rebuild
        as `[Bid: <qty>, <face>]`.
        """
        if isinstance(matched, tuple):
            if env_id == "LiarsDice-v0":
                q, f = matched[0].strip(), matched[1].strip()
                if q == "" and f == "":
                    return "[Call]"
                return f"[Bid: {q}, {f}]"
            # Per-env separator between tuple groups:
            #   Breakthrough UCI:  no separator  -> [a2a3]
            #   Othello:           comma+space   -> [2, 3]
            #   Nim / UTTT:        single space  -> [0 3]
            if env_id == "Breakthrough-v0":
                sep = ""
            elif env_id == "Othello-v0":
                sep = ", "
            else:
                sep = " "
            return "[" + sep.join(matched).strip() + "]"
        if matched.startswith("["):
            return matched
        return f"[{matched}]"

    # Compute legal action set once. We need it for step 1's guard
    # (reject "Action: <illegal>") as well as step 2.
    legal_actions: List[str] = []
    if observation is not None:
        legal_actions = extract_actions_from_observation(observation, env_id)

    def _iter_pattern_matches():
        """Yield every `_candidate(...)`-built action string found in
        `text` via `pattern`, in order of appearance. Each match's
        groups (or whole match) are passed through `_candidate` so the
        rebuilt string matches the env's expected token shape."""
        for m in re.finditer(pattern, text):
            groups = m.groups()
            if any(g is not None for g in groups):
                yield _candidate(groups)
            else:
                yield _candidate(m.group(0))

    # 1) "Action: <token>" prefix (any format the env's pattern matches).
    #    If a legal-action set is available, require the extracted
    #    token to be in it; this rejects LLM outputs that try to move
    #    the wrong side (a common Breakthrough failure that ends the
    #    game early). When no legal set is available (e.g. the env
    #    doesn't publish legal moves), accept the literal token.
    m = re.search(
        r"Action\s*:\s*(" + pattern + r")",
        text, re.IGNORECASE,
    )
    if m:
        cand = _candidate(m.group(1))
        if not legal_actions or cand in legal_actions:
            return cand
        # Else: the LLM's stated action is illegal; fall through to
        # step 2 which may find a legal alternative in the prose.

    # 2) First match of `move_pattern` in the output that is in legal set.
    if legal_actions:
        for cand in _iter_pattern_matches():
            if cand in legal_actions:
                return cand

    # 3) Any match of `move_pattern` in the output. If a legal set is
    #    available but the LLM mentioned NO legal token (we reach here
    #    because step 2 found nothing), returning the LLM's illegal
    #    token would — for games like Breakthrough — end the game
    #    prematurely. In that case prefer the first legal action as a
    #    safe default. Only when no legal set is available do we trust
    #    the LLM's literal token.
    first = re.search(pattern, text)
    if first:
        if legal_actions:
            # The LLM's output contains a pattern-shaped token, but
            # none of them were legal (step 2 exhausted them). Use a
            # legal default rather than feeding the env an illegal
            # move that could terminate the game.
            return legal_actions[0]
        groups = first.groups()
        if any(g is not None for g in groups):
            return _candidate(groups)
        return _candidate(first.group(0))

    # 4) Strip common leading phrases and use remainder.
    cleaned = re.sub(
        r"^\s*(action|answer|reply|response|output)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or text


# ---------------------------------------------------------------------------
# Game result / env wrapper
# ---------------------------------------------------------------------------
@dataclass
class GameResult:
    """Result from a game played in TextArena."""
    rewards: List[float]
    game_info: Dict
    done: bool
    player_ids: List[int]
    trajectory: List[Dict] = None  # List of {"player_id": int, "observation": str, "action": str}

    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []

    @property
    def reward_dict(self) -> Dict:
        """Get rewards as dict (for backward compatibility)."""
        return self.game_info.get("rewards", {})


class TextArenaEnv:
    """Wrapper for TextArena environments."""

    def __init__(
        self,
        env_id: str = "TicTacToe-v0",
        num_players: int = 2,
        agent_name: str = "openrouter/gpt-4o-mini"
    ):
        self.env_id = env_id
        self.num_players = num_players
        self.agent_name = agent_name
        self.env: Optional[ta.Env] = None
        self.current_player: int = 0

    def reset(self) -> Tuple[int, str]:
        """Reset the environment and return initial observation."""
        self.env = ta.make(self.env_id)
        self.env.reset(num_players=self.num_players)
        self.current_player = 0
        return self.env.get_observation()

    def step(self, action: str) -> GameResult:
        """Execute an action and return the result."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        done, step_info = self.env.step(action=action)
        return GameResult(
            rewards=step_info.get("rewards", []),
            game_info=step_info,
            done=done,
            player_ids=[p for p in range(self.num_players)]
        )

    def get_observation(self) -> Tuple[int, str]:
        """Get current player's observation."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return self.env.get_observation()

    def close(self) -> Tuple[List[float], Dict]:
        """Close the environment and get final rewards."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return self.env.close()

    @classmethod
    def _observation_tail(cls, observation: str, env_id: Optional[str] = None) -> str:
        """
        Return the "live" tail of a TextArena observation.

        The tail-extraction strategy is registered per env in GAME_SPECS
        (e.g. TicTacToe uses "[GAME] Current Board:" while ConnectFour uses
        "[GAME] Board state:"). When `env_id` is None we fall back to
        the default extractor, which handles the common cases.
        """
        spec = get_action_format(env_id) if env_id else DEFAULT_GAME_SPEC
        tail = spec["tail_extractor"](observation)
        if not tail:
            # Fallback: keep the last 300 chars as a reasonable summary.
            return observation[-300:] if len(observation) > 300 else observation
        return tail

    def play_game(
        self,
        agents: List[callable],
        verbose: bool = False,
        record_trajectory: bool = True,
        brs_player_id: Optional[int] = None,
        player_roles: Optional[Dict[int, str]] = None,
    ) -> GameResult:
        """
        Play a complete game with given agents.

        TextArena returns a cumulative observation each turn — it contains
        the full transcript of every prior action, viewed from the current
        player's perspective. We record only the "live" tail (the part
        starting at the latest "[GAME] Current Board:" snapshot) so the
        trajectory doesn't bloat from duplicated history. Each step is
        recorded as:

            - step: 0-indexed turn number
            - player_id: who acted
            - action: the action they took
            - observation: the *tail* of the TextArena observation
              (board state + Available Moves for the acting player). The
              field is named "observation" for ergonomic reads even
              though it is the trimmed tail, not the raw TextArena
              observation. To get the current legal moves, call
              `extract_actions_from_observation(entry["observation"], env_id)`.
            - is_brs_step: True if the player who acted is the BRS
              strategy (only set when brs_player_id is provided).
            - role: semantic role of the player who acted (e.g. "brs",
              "opponent"). Only set when player_roles is provided.

        Args:
            agents: List of agent functions that take observation and return action
            verbose: Whether to print game progress
            record_trajectory: Whether to record observations and actions
            brs_player_id: Which player slot holds the BRS strategy. When
                set, every trajectory entry gains an `is_brs_step` bool so
                downstream analysis (e.g. diagnosing "did BRS make a bad
                move?") can filter self vs opponent steps without re-deriving
                from the meta-strategy. If None, the field is omitted.
            player_roles: Optional {player_id: role_string} mapping for
                richer per-step labelling. Each trajectory entry gains a
                `role` field whose value is `player_roles[player_id]` (or
                None if not mapped).

        Returns:
            GameResult with final rewards, game info, and trajectory
        """
        player_id, observation = self.reset()

        # print('='*50)
        # print(player_id)
        # print('='*50)

        trajectory: List[Dict] = []

        if verbose:
            print(f"Starting game: {self.env_id}")

        step_count = 0
        max_steps = 100  # Safety limit

        while step_count < max_steps:
            step_count += 1

            if player_id >= len(agents):
                raise ValueError(f"Player {player_id} has no agent")

            action = agents[player_id](observation)

            # print(observation)


            if record_trajectory:
                entry: Dict = {
                    "step": step_count - 1,
                    "player_id": player_id,
                    "action": action,
                    # The "live" tail of the cumulative observation
                    # (from the latest board snapshot; the exact anchor
                    # is per-env, e.g. "[GAME] Current Board:" for
                    # TicTacToe, "[GAME] Board state:" for ConnectFour).
                    # The field is named "observation" for ergonomic
                    # reads, even though it is the trimmed tail, not
                    # the raw TextArena observation. Use
                    # `extract_actions_from_observation(entry["observation"], env_id)`
                    # to get the current legal moves.
                    "observation": self._observation_tail(observation, env_id=self.env_id),
                }
                if brs_player_id is not None:
                    entry["is_brs_step"] = (player_id == brs_player_id)
                if player_roles is not None:
                    entry["role"] = player_roles.get(player_id)
                trajectory.append(entry)

            if verbose:
                print(f"Player {player_id}: {action[:50]}...")

            done, step_info = self.env.step(action=action)

            if done:
                final_rewards, game_info = self.env.close()
                # Normalize rewards to list format
                if isinstance(final_rewards, dict):
                    rewards_list = [final_rewards.get(i, 0.0) for i in range(self.num_players)]
                else:
                    rewards_list = final_rewards if isinstance(final_rewards, list) else [final_rewards]
                return GameResult(
                    rewards=rewards_list,
                    game_info=game_info,
                    done=True,
                    player_ids=[p for p in range(self.num_players)],
                    trajectory=trajectory
                )

            player_id, observation = self.env.get_observation()

        # A harness timeout is not a game-defined terminal outcome.
        final_rewards, game_info = self.env.close()
        if isinstance(final_rewards, dict):
            rewards_list = [final_rewards.get(i, 0.0) for i in range(self.num_players)]
        else:
            rewards_list = final_rewards if isinstance(final_rewards, list) else [final_rewards]
        return GameResult(
            rewards=rewards_list,
            game_info=game_info,
            done=False,
            player_ids=[p for p in range(self.num_players)],
            trajectory=trajectory
        )

    @staticmethod
    def get_available_games() -> List[str]:
        """Get list of available game IDs."""
        # TextArena provides 100+ games
        # Common games for testing:
        return [
            "TicTacToe-v0",
            "ConnectFour-v0",
            "2048-v0",
            # Add more as needed
        ]
