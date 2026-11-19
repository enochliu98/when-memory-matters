"""
Memory Items for the LangPSRO Bench.

Each Memory Item is a piece of auxiliary information (a Trajectory,
Insight, or Reasoning Pattern) that the agent sees baked into its
`### Language Policy` (i.e. the standing strategy). v0 supports two
kinds of items:

  - Trajectory: 1 win game + 1 loss game, natural-language walkthrough
    with [N] cell/column tags (case-based)
  - Insight: a free-form strategy text (e.g. "win first, block next")

Memory Item data is stored as Python dicts (not JSON) so editors can
type-check and so the rendering helpers can live next to the data.
"""

from typing import Literal, TypedDict, List, Dict, Optional


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------
MemoryItemType = Literal["trajectory", "insight", "reasoning_pattern"]


class TrajectoryGame(TypedDict):
    outcome: Literal["WIN", "LOSS"]
    opponent: str          # free-form label, e.g. "self-play greedy"
    narrative: str         # natural-language walkthrough with [N] cell/column tags
    lesson: str            # one-sentence takeaway


class TrajectoryItem(TypedDict):
    type: Literal["trajectory"]
    env_id: str
    description: str
    games: List[TrajectoryGame]   # length 2 for v0: [win, loss]


class InsightItem(TypedDict):
    type: Literal["insight"]
    env_id: str
    description: str
    insight_text: str      # free-form strategy text (the "Insight" itself)


# ---------------------------------------------------------------------------
# Trajectory items
# ---------------------------------------------------------------------------
# Cell numbering reminder:
#   TicTacToe cells:
#       0 | 1 | 2
#       ---------
#       3 | 4 | 5
#       ---------
#       6 | 7 | 8
#
#   ConnectFour columns: 0..6 left to right, bottom row first (cell 0
#   is the bottom of column 0, etc.). The "column number" used in
#   actions like "[3]" is the column index.

_TICTACTOE_TRAJECTORY: TrajectoryItem = {
    "type": "trajectory",
    "env_id": "TicTacToe-v0",
    "description": (
        "Two TicTacToe games: one won by playing the center + opposite "
        "corner + opposite-side fork pattern, one lost by opening a corner."
    ),
    "games": [
        {
            "outcome": "WIN",
            "opponent": "self-play baseline",
            "narrative": (
                "I opened center [4] (it controls the most lines). "
                "Opponent took corner [0]. "
                "I took the opposite corner [8] to set up a diagonal threat. "
                "Opponent played side [1] to block. "
                "I took side [7] — this gives me two winning threats at once: "
                "row [6]-[7]-[8] only needs one more, and the diagonal "
                "[2]-[4]-[6] is also brewing. "
                "Opponent was forced to block at [6]. "
                "I took the remaining corner [2] and won on the diagonal."
            ),
            "lesson": (
                "Open center, mirror with opposite corner to threaten two "
                "lines, then take the opposite side of opponent's block to "
                "create an unblockable fork."
            ),
        },
        {
            "outcome": "LOSS",
            "opponent": "self-play baseline",
            "narrative": (
                "I opened corner [0]. "
                "Opponent replied center [4], claiming both diagonals. "
                "I tried side [1], opponent extended with [7] threatening "
                "the bottom row. "
                "I blocked at [3]. "
                "Opponent played [5] and now threatened the right column. "
                "I missed it and played [2]. "
                "Opponent completed [2]-[4]-[6] diagonal to win."
            ),
            "lesson": (
                "Opening a corner (instead of center) lets opponent take "
                "the center and forces you to defend two diagonals. "
                "Open center, or — if center is taken — pick the corner "
                "that gives you the most unblocked lines."
            ),
        },
    ],
}


_CONNECTFOUR_TRAJECTORY: TrajectoryItem = {
    "type": "trajectory",
    "env_id": "ConnectFour-v0",
    "description": (
        "Two ConnectFour games: one won by stacking in opponent's column, "
        "one lost by failing to match the opponent's column stack."
    ),
    "games": [
        {
            "outcome": "WIN",
            "opponent": "self-play baseline",
            "narrative": (
                "Opponent opened column [3]. "
                "I replied [3] (stacked directly above). "
                "Opponent played [2]. "
                "I extended my column with [3] again — now I have three "
                "in column [3] (vertical-3). "
                "Opponent took [4]. "
                "I played [3] to complete the vertical four and won."
            ),
            "lesson": (
                "When the opponent opens a center-ish column, mirror by "
                "stacking in the SAME column to build a vertical four. "
                "Don't spread your pieces across columns unless you have "
                "an explicit horizontal or diagonal threat."
            ),
        },
        {
            "outcome": "LOSS",
            "opponent": "self-play baseline",
            "narrative": (
                "Opponent opened column [3]. "
                "I played [0] (spreading out). "
                "Opponent [3] (their second in column 3). "
                "I played [1]. "
                "Opponent [3] (third in column 3). "
                "I played [2]. "
                "Opponent [3] completes vertical four — I lost. "
                "I should have stacked at [3] from move 2 to block their "
                "vertical build."
            ),
            "lesson": (
                "When an opponent is stacking in one column, you MUST match "
                "the stack tempo (play directly above their last move) — "
                "never let them get three in a column unchallenged."
            ),
        },
    ],
}


# Registry: env_id -> list of items available for that env. v0 has one
# Trajectory item per env (the win+loss pair).
TRAJECTORY_ITEMS: Dict[str, List[TrajectoryItem]] = {
    "TicTacToe-v0": [_TICTACTOE_TRAJECTORY],
    "ConnectFour-v0": [_CONNECTFOUR_TRAJECTORY],
}


# ---------------------------------------------------------------------------
# Insight items
# ---------------------------------------------------------------------------
# Free-form strategy text the agent should follow. Unlike Trajectory
# items (which are concrete game walkthroughs), Insight items are
# abstract principles or rules distilled from experience.

_TICTACTOE_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "TicTacToe-v0",
    "description": (
        "Three-rule TicTacToe insight covering positional value, the "
        "Fork tactic, and active defense. Tests whether the LLM can "
        "absorb and apply a denser rule set than a single win/block "
        "heuristic."
    ),
    "insight_text": (
        "You are a Tic Tac Toe player. Apply the following principles "
        "to maximize your chance of winning:\n\n"
        "1. Positional Hierarchy: The Center is the most valuable square "
        "(involved in 4 winning lines), followed by Corners (3 winning "
        "lines), and finally Edges (2 winning lines). Prefer higher-value "
        "squares when no immediate threat is present.\n\n"
        "2. The 'Fork' Tactic: The ultimate offensive strategy is creating "
        "a 'fork' — a placement that simultaneously creates two distinct, "
        "non-overlapping winning threats, making it impossible for the "
        "opponent to defend both. After you control the center, look for "
        "fork opportunities every turn.\n\n"
        "3. Active Defense: The safest way to defend against an opponent's "
        "potential fork is to go on the offensive. Create an immediate "
        "two-in-a-row threat that forces the opponent to block, "
        "disrupting their setup (ensure your threat doesn't force them "
        "into the very square they need to fork).\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_CONNECTFOUR_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "ConnectFour-v0",
    "description": (
        "Four-rule ConnectFour insight covering center-column dominance, "
        "gravity / 'The Claim', multi-threat (forks & traps), and the "
        "parity concept. Tests whether the LLM can apply a denser, "
        "ConnectFour-specific rule set than a generic win/block heuristic."
    ),
    "insight_text": (
        "You are a Connect Four player. Apply the following principles "
        "to maximize your chance of winning:\n\n"
        "1. Center Column Dominance: The center column (Column 4) is the "
        "absolute most critical area of the board. A piece in the center "
        "column participates in the maximum possible number of "
        "horizontal and diagonal winning lines. The closer a piece is "
        "to the center, the higher its strategic value.\n\n"
        "2. Gravity and 'The Claim': Because pieces fall to the lowest "
        "available slot, you do not have access to the whole board. You "
        "must constantly evaluate the slot directly above where you "
        "intend to play. Playing a careless move can act as a stepping "
        "stone, handing your opponent an immediate win on top of your "
        "piece.\n\n"
        "3. Multi-Threats (Forks & Traps): The ultimate offensive goal "
        "is to create multiple, non-overlapping 3-in-a-row threats "
        "(e.g., the classic '7-trap' or horizontal/diagonal "
        "intersections). If you have two winning spaces but the "
        "opponent can only play one piece per turn, you guarantee a "
        "win.\n\n"
        "4. Parity Concept: In the endgame, row height matters. "
        "Threats on odd rows (1, 3, 5 counting from the bottom) "
        "naturally favor the first player, while threats on even rows "
        "(2, 4, 6) favor the second player due to the alternating "
        "turn order and fixed column capacities.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_NIM_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "Nim-v0",
    "description": (
        "Three-rule Nim insight covering the Nim-Sum (XOR) operation, "
        "the 'zero is the goal' invariant, and the winning tactic of "
        "always reducing the new Nim-sum to 0. Tests whether the LLM "
        "can absorb and apply a math-driven rule set (vs. a heuristic "
        "rule set like TTT/C4)."
    ),
    "insight_text": (
        "You are a Nim player. Apply the following principles to "
        "maximize your chance of winning:\n\n"
        "1. The Nim-Sum (XOR): The ultimate strategy of Nim relies "
        "entirely on the bitwise XOR operation. The 'Nim-sum' is "
        "calculated by taking the XOR sum of the number of objects "
        "in all remaining heaps.\n\n"
        "2. Zero is the Goal: A board state with a Nim-sum of 0 is a "
        "'losing state' for whoever must move next. A state with a "
        "Nim-sum greater than 0 is a 'winning state.'\n\n"
        "3. The Winning Tactic: If the current Nim-sum is greater "
        "than 0, there is always at least one valid move that will "
        "reduce the new Nim-sum to exactly 0. Your objective is to "
        "always leave a Nim-sum of 0 for your opponent.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_UTTT_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "UltimateTicTacToe-v0",
    "description": (
        "Four-rule Ultimate Tic-Tac-Toe insight covering the Pointer "
        "Mechanic, the Free Move disaster, global Center Board "
        "Dominance, and the Sacrifice-for-Position concept. Tests "
        "whether the LLM can reason about meta-strategy (controlling "
        "opponent's future moves) rather than just local wins."
    ),
    "insight_text": (
        "You are an Ultimate Tic-Tac-Toe player. Apply the following "
        "principles to maximize your chance of winning:\n\n"
        "1. The Pointer Mechanic (The Golden Rule): The game is "
        "about controlling your opponent. The local relative "
        "coordinate of your move exactly determines the global board "
        "where your opponent must play their next turn. Spatial "
        "control is vastly more important than blindly winning a "
        "local board.\n\n"
        "2. The 'Free Move' Disaster: If you send your opponent to "
        "a local board that is already won or completely full, they "
        "are granted a 'Free Move' - the ability to play on ANY "
        "available square on the entire grid. Handing your opponent a "
        "Free Move is extremely dangerous and often game-losing. "
        "Avoid this at all costs.\n\n"
        "3. Center Board Dominance: The global center board is "
        "highly contested because it participates in 4 global "
        "winning lines. Consequently, playing in the center square "
        "of any local board (which sends the opponent to the center "
        "board) is a powerful double-edged sword. You must evaluate "
        "if the opponent has an advantage in the center board "
        "before sending them there.\n\n"
        "4. Sacrifice for Position: Sometimes, it is strategically "
        "correct to intentionally lose a local board if blocking the "
        "opponent would force you to play on a square that gives "
        "them a Free Move or sends them to a critical global board.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_OTHELLO_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "Othello-v0",
    "description": (
        "Five-rule Othello/Reversi insight covering the Disc Count "
        "Fallacy, Mobility Mastery, Absolute Stability (corners), "
        "Danger Zones (X- and C-squares), and Quiet Moves. Tests "
        "whether the LLM can reason about board control and mobility "
        "(not just disc count) and avoid common Othello pitfalls "
        "like greedily flipping many discs early."
    ),
    "insight_text": (
        "You are an Othello (Reversi) player. Apply the following "
        "principles to maximize your chance of winning:\n\n"
        "1. The Disc Count Fallacy: Novices try to flip as many "
        "discs as possible in the early and mid-game. This is a "
        "fatal flaw. Having too many discs early on increases your "
        "'frontier' (discs touching empty spaces), which gives your "
        "opponent more options to flank you. Focus on board control, "
        "not disc count, until the final endgame.\n\n"
        "2. Mobility is Mastery: The primary strategic goal is to "
        "restrict your opponent's valid moves while maximizing your "
        "own. If you force your opponent into a situation where they "
        "only have one or two terrible moves, you dictate the flow "
        "of the game.\n\n"
        "3. Absolute Stability (Corners): Corner squares can never be "
        "outflanked or flipped. They are permanently yours. Securing "
        "a corner allows you to build stable, unflippable edges "
        "radiating from it.\n\n"
        "4. The Danger Zones (X and C Squares): Squares strictly "
        "diagonal to an empty corner are called 'X-squares.' Squares "
        "directly horizontal or vertical to an empty corner are "
        "'C-squares.' Playing on these squares usually hands the "
        "adjacent corner directly to your opponent.\n\n"
        "5. Quiet Moves: A good mid-game move is often a 'quiet "
        "move' - a move that flips only a few discs, and importantly, "
        "flips 'internal' discs (discs completely surrounded by "
        "others) rather than frontier discs.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_BREAKTHROUGH_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "Breakthrough-v0",
    "description": (
        "Four-rule Breakthrough insight covering the Unstoppable "
        "Runner concept, Pawn Chains (Phalanx), Defensive Integrity, "
        "and the Decoy Sacrifice. Tests whether the LLM can reason "
        "about long-term positional races (a runner to the back row) "
        "vs. just local capture tactics."
    ),
    "insight_text": (
        "You are a Breakthrough player. Apply the following "
        "principles to maximize your chance of winning:\n\n"
        "1. The Unstoppable Runner: The ultimate goal is to create a "
        "'runner' - a piece that has bypassed all enemy defenders in "
        "its column and adjacent columns. Once a runner has a clear, "
        "unchallenged path to the opposite end, the game becomes a "
        "pure race.\n\n"
        "2. Pawn Chains (The Phalanx): Pieces are strongest when "
        "arranged in diagonal chains. A piece backed by another "
        "friendly piece diagonally is 'protected,' because if it gets "
        "captured, the supporting piece can immediately move forward "
        "to recapture or take its place.\n\n"
        "3. Defensive Integrity: Advancing too rapidly or moving "
        "multiple pieces from the same file creates fatal gaps in "
        "your defense. You must maintain a solid rear guard to "
        "prevent the opponent's pieces from slipping through your "
        "lines.\n\n"
        "4. The Decoy Sacrifice: Because captures are mandatory when "
        "a piece occupies a diagonal path you wish to take, you can "
        "intentionally sacrifice a piece to pull a crucial enemy "
        "defender out of its column, thereby opening a clear lane "
        "for your actual runner.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_LIARSDICE_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "LiarsDice-v0",
    "description": (
        "Four-rule Liar's Dice insight covering the EV anchor (1/3 per "
        "unknown die with '1's wild), anchoring decisions to your own "
        "hand, the information economy of bids, and the powerful Pivot "
        "tactic (raise face, keep quantity). Tests whether the LLM can "
        "absorb and apply probability-anchored reasoning rules to a "
        "hidden-info game (vs. the deterministic board-game tactics in "
        "TTT/C4/Nim)."
    ),
    "insight_text": (
        "You are a Liar's Dice player. Apply the following principles "
        "to maximize your chance of winning:\n\n"
        "1. The Expected Value (EV) Math: Assuming '1's are wildcards, "
        "any unknown die has a 1/3 chance of matching a specific face "
        "value (e.g. a '4' or a '1'). Therefore, the baseline expected "
        "number of a specific face value among all UNKNOWN dice is "
        "(Total Unknown Dice) / 3. Use this as your anchor for "
        "evaluating every bid and every Call decision.\n\n"
        "2. Your Hand as the Anchor: Your own dice are the only "
        "absolute truth on the table. Your strategic advantage comes "
        "from the deviation in your hand. If you hold three '5's out "
        "of your 5 dice, the table is massively skewed toward '5's, "
        "making a high bid on '5's very safe.\n\n"
        "3. The Information Economy: Every bid sends a signal. If an "
        "opponent confidently raises the quantity of '4's, they likely "
        "hold multiple '4's (or '1's). Conversely, if you bid high on "
        "a number you do NOT have, you are setting a trap, but you "
        "risk being called.\n\n"
        "4. The Pivot: You don't always have to raise the quantity. "
        "You can raise the face value while keeping the quantity the "
        "same (e.g. bidding four '6's over four '3's). This is a "
        "powerful tool to shift the battleground to the numbers you "
        "actually hold in your hand.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


_KUHNPOKER_INSIGHT: InsightItem = {
    "type": "insight",
    "env_id": "KuhnPoker-v0",
    "description": (
        "Four-rule Kuhn Poker insight covering the deck/absolute strength, "
        "the Game Theory Optimal (GTO) requirement for mixed strategies, "
        "the role of each card (K for value/trap, J for bluff, Q for "
        "bluff-catching), and the math of balancing value bets with "
        "bluffs. Tests whether the LLM can reason about hidden-info "
        "games and randomized play rather than deterministic board-game "
        "tactics."
    ),
    "insight_text": (
        "You are a Kuhn Poker player. Apply the following principles "
        "to maximize your chance of winning:\n\n"
        "1. The Deck & Absolute Strength: The game uses only 3 cards. "
        "Let's assume standard ranks: King (K, the nuts, always wins), "
        "Queen (Q, middle card), and Jack (J, the worst, always "
        "loses).\n\n"
        "2. Game Theory Optimal (GTO): Unlike chess or board games, "
        "playing predictably in poker is fatal. You cannot win by only "
        "betting your Kings and folding your Jacks. You must use a "
        "'Mixed Strategy' (randomizing certain actions) to remain "
        "unexploitable.\n\n"
        "3. The Roles of the Cards:\n"
        "   - King (K) is for Value Betting (betting to get called by "
        "a worse card) and Trapping (checking to induce a bluff).\n"
        "   - Jack (J) is for Bluffing (betting to force a Queen to "
        "fold) or giving up. You never call a bet with a Jack.\n"
        "   - Queen (Q) is a Bluff-Catcher. It cannot value bet (a "
        "Jack will fold, a King will call/win). Its only purpose is to "
        "catch the opponent bluffing with a Jack, or to check down to "
        "showdown.\n\n"
        "4. The Math of Balancing: When you bet, your betting range "
        "must be balanced between value bets (Kings) and bluffs "
        "(Jacks) so the opponent is mathematically indifferent to "
        "calling or folding with their Queens.\n\n"
        "Remember: Your goal is to WIN."
    ),
}


INSIGHT_ITEMS: Dict[str, List[InsightItem]] = {
    "TicTacToe-v0": [_TICTACTOE_INSIGHT],
    "ConnectFour-v0": [_CONNECTFOUR_INSIGHT],
    "Nim-v0": [_NIM_INSIGHT],
    "UltimateTicTacToe-v0": [_UTTT_INSIGHT],
    "Othello-v0": [_OTHELLO_INSIGHT],
    "Breakthrough-v0": [_BREAKTHROUGH_INSIGHT],
    "KuhnPoker-v0": [_KUHNPOKER_INSIGHT],
    "LiarsDice-v0": [_LIARSDICE_INSIGHT],
}


# ---------------------------------------------------------------------------
# Reasoning Pattern items
# ---------------------------------------------------------------------------
# Ordered checklist (or decision template) the agent runs through
# before each move. The pattern is meant to be applied step by step,
# not learned as free-form prose.

class ReasoningPatternItem(TypedDict):
    type: Literal["reasoning_pattern"]
    env_id: str
    description: str
    pattern_text: str      # ordered checklist the agent should follow


_LIARSDICE_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "LiarsDice-v0",
    "description": (
        "5-step Liar's Dice decision checklist (Calculate Ground Truth "
        "-> Liar Threshold -> Pivot -> Value Push -> Calculated Bluff / "
        "Survival). The checklist is strictly prioritized and forced to "
        "output an EV calculation in Step 1 before any action. Tests "
        "whether the LLM can reason step-by-step under probabilistic "
        "uncertainty and execute the FIRST applicable condition."
    ),
    "pattern_text": (
        "When it is your turn, evaluate the current bid and your hand "
        "by strictly moving down this prioritized mathematical "
        "checklist. Execute the first condition that applies:\n\n"
        "1. Calculate the Ground Truth:\n"
        "   - Count your own dice that match the current bid's face "
        "value (Remember: count '1's as matches!).\n"
        "   - Calculate the Expected Value (EV) of the unknown dice: "
        "(Total Dice in Game - Your Dice) / 3.\n"
        "   - Calculate the 'Total Expected Quantity' = Your Matches + "
        "EV.\n\n"
        "2. The 'Liar' Threshold (Challenge): Is the current bid's "
        "quantity strictly greater than your 'Total Expected Quantity' "
        "+ 1? (e.g. they bid 7, but the math says expect 5). If yes, "
        "the bid is statistically highly improbable. Immediately call "
        "'Liar' (Challenge) by typing '[Call]'.\n\n"
        "3. The Pivot (Safe Shift): If the current bid is safe, do you "
        "hold a strong cluster of a DIFFERENT face value (e.g. you hold "
        "multiple '5's, but the current bid is on '2's)? If yes, bid "
        "the same quantity (or quantity + 1) but change the face value "
        "to your strongest number to safely pass the turn.\n\n"
        "4. The Value Push (Safe Raise): Do you hold a significant "
        "number of matches for the CURRENT bid's face value? If yes, "
        "confidently increase the quantity by exactly 1.\n\n"
        "5. The Calculated Bluff / Survival: If the current bid is "
        "statistically safe (meaning you shouldn't call 'Liar') but you "
        "hold ZERO matches to help, you must bluff to survive.\n"
        "   - Either raise the face value to the absolute highest "
        "possible (e.g. '6's) without raising the quantity, forcing the "
        "next player into a mathematical corner.\n"
        "   - Or raise the quantity by 1 on a face value where you at "
        "least have ONE die, minimizing your exposure.\n\n"
        "Action Required: Before making your decision, output a brief "
        "calculation of the EV (e.g. 'Step 1: Current bid is five '4's. "
        "I have two '4's and one '1' (3 matches). Unknown dice = 10. EV "
        "of unknown = 10 / 3 = 3.33. Total expected = 6.33. Step 2: The "
        "bid of 5 is safe. Step 3: I have a lot of 4s, I will use Step "
        "4 and push the value.'). Then output your final action: "
        "either your new bid (Quantity and Face Value) or 'Call Liar'."
    ),
}


_TICTACTOE_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "TicTacToe-v0",
    "description": (
        "8-step TicTacToe decision checklist (Win -> Block -> Fork -> "
        "Block Fork -> Center -> Opposite Corner -> Empty Corner -> "
        "Empty Edge). Tests whether the LLM can follow a strict "
        "prioritized template step by step."
    ),
    "pattern_text": (
        "When it is your turn, evaluate the board state by strictly "
        "moving down this prioritized checklist. Execute the first "
        "condition that is met and immediately stop evaluating:\n\n"
        "1. Win: Do you have two in a row with an empty third square? "
        "If yes, play that square to win.\n"
        "2. Block: Does the opponent have two in a row with an empty "
        "third square? If yes, play that square to block them.\n"
        "3. Fork: Can you play a square that creates two distinct "
        "winning threats for your next turn? If yes, play it.\n"
        "4. Block Fork: Can the opponent create a fork on their next "
        "turn? If yes, proactively block it by either:\n"
        "   - Creating a two-in-a-row threat to force them to defend "
        "elsewhere.\n"
        "   - Occupying the exact intersection they need to complete "
        "their fork.\n"
        "5. Center: Is the center square empty? If yes, play it.\n"
        "6. Opposite Corner: Is the opponent in a corner square, and is "
        "the diagonally opposite corner empty? If yes, play the "
        "opposite corner.\n"
        "7. Empty Corner: Is there any empty corner square available? "
        "If yes, play it.\n"
        "8. Empty Edge: Is there any empty edge square available? If "
        "yes, play it.\n\n"
    ),
}


_CONNECTFOUR_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "ConnectFour-v0",
    "description": (
        "7-step ConnectFour decision checklist (Immediate Win -> "
        "Immediate Block -> Safety Check -> Create a Fork -> Block "
        "Opponent's Setup -> Center Control & Proximity -> Extend "
        "Connectivity). Tests whether the LLM can follow a strict "
        "prioritized template, including a per-column safety check, "
        "before picking a column."
    ),
    "pattern_text": (
        "When it is your turn, evaluate the board state by strictly "
        "moving down this prioritized checklist. A standard Connect 4 "
        "board has 7 columns (1-7) and 6 rows. Execute the first "
        "condition that is met and immediately stop evaluating:\n\n"
        "1. Immediate Win: Do you have a 3-in-a-row (horizontal, "
        "vertical, or diagonal) where the winning 4th slot is "
        "currently supported and playable? If yes, drop your piece in "
        "that column to win.\n"
        "2. Immediate Block: Does the opponent have a 3-in-a-row with "
        "a supported, playable 4th slot? If yes, you must drop your "
        "piece in that column to block them.\n"
        "3. Safety Check (Crucial): Before considering any offensive "
        "moves below, analyze all 7 columns. If dropping a piece in "
        "Column X creates a playable slot directly above it that "
        "allows your opponent to win on their next turn, eliminate "
        "Column X from your current options (unless forced).\n"
        "4. Create a Fork (Double Threat): Among the safe columns, can "
        "you drop a piece that simultaneously creates two distinct "
        "winning threats for the future? If yes, play it.\n"
        "5. Block Opponent's Setup: Among the safe columns, is the "
        "opponent building an open-ended horizontal 2-in-a-row (which "
        "will become an unblockable 3-in-a-row next turn), or setting "
        "up a multi-directional trap? If yes, drop a piece to disrupt "
        "their formation.\n"
        "6. Center Control & Proximity: If no immediate tactical "
        "threats exist, prioritize playing in the center. Choose the "
        "safest available column closest to the center in this order "
        "of preference: Column 4 > Columns 3 & 5 > Columns 2 & 6 > "
        "Columns 1 & 7.\n"
        "7. Extend Connectivity: When choosing between equally central "
        "columns, select the drop that connects to your existing "
        "pieces, maximizing the number of potential overlapping lines "
        "rather than placing an isolated piece.\n\n"
        "Action Required: Before making your move, output a brief "
        "step-by-step evaluation of this checklist (e.g., 'Step 1: No. "
        "Step 2: No. Step 3: Columns 2 and 5 are unsafe. Step 4...'), "
        "and then output your final chosen column number (1-7)."
    ),
}


_NIM_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "Nim-v0",
    "description": (
        "3-step Nim decision checklist: Calculate the current Nim-sum "
        "(XOR of all heap sizes) -> if > 0, find a heap where "
        "size XOR nim-sum < size and reduce it -> if nim-sum is 0, "
        "stall by removing 1 from the largest heap. Tests whether the "
        "LLM can do bitwise XOR reasoning step by step."
    ),
    "pattern_text": (
        "When it is your turn, evaluate the board state by following "
        "this high-level sequence. Execute the first condition that "
        "applies:\n\n"
        "1. Calculate State: First, determine the current Nim-sum by "
        "calculating the XOR sum of all remaining heap sizes.\n"
        "2. Winning Move Search: If the current Nim-sum is greater "
        "than 0, identify the target heap. Find a heap where its size "
        "XOR the current Nim-sum is strictly less than its current "
        "size. Reduce this heap to the new calculated size, which "
        "forces the new board Nim-sum to 0.\n"
        "3. Survival Move: If the current Nim-sum is already 0 (you "
        "are in a mathematically losing position), you cannot force a "
        "win. Remove exactly 1 item from the largest available heap "
        "to stall and wait for the opponent to make a calculation "
        "error.\n\n"
        "Action Required: Before making your move, output a brief "
        "summary of your Nim-sum calculation and identify which "
        "condition applies, then output your final chosen move (which "
        "heap to target and how many items to remove)."
    ),
}


_UTTT_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "UltimateTicTacToe-v0",
    "description": (
        "7-step Ultimate Tic-Tac-Toe decision checklist (Ultimate Win "
        "-> Safety Filter -> Safe Local Win -> Strategic Block -> "
        "Send to a 'Dead' or 'Harmless' Board -> Build Local Threats -> "
        "Positional Play). The Safety Filter (Step 2) is the key "
        "step that eliminates squares which would hand the opponent a "
        "Free Move. Tests whether the LLM can apply a multi-step "
        "checklist under strict elimination rules."
    ),
    "pattern_text": (
        "When it is your turn, you are either constrained to a "
        "specific local board or you have a Free Move. Evaluate the "
        "board state and your legal moves by strictly moving down "
        "this prioritized checklist. Execute the first condition "
        "that is met:\n\n"
        "1. Ultimate Win: Can you play a square that wins the "
        "current local board AND simultaneously completes a "
        "3-in-a-row on the global board to win the entire game? If "
        "yes, play it.\n"
        "2. Safety Filter (Crucial): Before making any offensive or "
        "defensive moves, identify all squares in your valid area "
        "that would send the opponent to a completed (won or full) "
        "local board. Eliminate these squares from your "
        "consideration completely, unless playing one directly wins "
        "the game.\n"
        "3. Safe Local Win: Among the safe squares remaining, does "
        "a move win the current local board, AND it does not send "
        "the opponent to a board where they are guaranteed a local "
        "win? If yes, play it.\n"
        "4. Strategic Block: Does the opponent have a 2-in-a-row on "
        "the current local board? If yes, block them ONLY IF the "
        "blocking square passes the Safety Filter (Step 2). If "
        "blocking gives them a Free Move, it is often better to let "
        "them win the local board.\n"
        "5. Send to a 'Dead' or 'Harmless' Board: Among the safe "
        "squares, actively choose a square that sends the opponent "
        "to a local board where they have zero advantage (e.g., an "
        "empty board, a board where you have the upper hand, or a "
        "board where they have no immediate 2-in-a-row threats).\n"
        "6. Build Local Threats: If the destination board is safe, "
        "look for a square that builds your own local 2-in-a-row "
        "(creating a local fork if possible).\n"
        "7. Positional Play: If all remaining safe options are equal, "
        "prioritize the local center square (if you control the "
        "global center board) or local corners.\n\n"
        "Action Required: Before making your move, output a brief "
        "step-by-step evaluation focusing especially on your 'Safety "
        "Filter' (e.g., 'Step 1: No. Step 2: Squares X and Y are "
        "eliminated because they give a Free Move. Step 3: No safe "
        "local win. Step 4: Blocking at Z is safe. I will play Z.'), "
        "and then output your final chosen coordinate."
    ),
}


_OTHELLO_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "Othello-v0",
    "description": (
        "5-step Othello decision checklist (Take the Corner -> "
        "Danger Zone Filter -> Exploit the Edge -> Minimize "
        "Opponent's Mobility -> The Quiet Move). The Danger Zone "
        "Filter (Step 2) eliminates X- and C-squares; mobility "
        "minimization (Step 4) and quiet moves (Step 5) are the "
        "core anti-fallacy heuristics. Tests whether the LLM can "
        "reason about board control rather than just disc count."
    ),
    "pattern_text": (
        "When it is your turn, evaluate your valid legal moves by "
        "strictly running them through this prioritized checklist. "
        "Execute the top-most applicable logic:\n\n"
        "1. Take the Corner (Absolute Priority): Is a corner square "
        "available as a valid move? If yes, take it immediately.\n"
        "2. Danger Zone Filter (Crucial): Identify all your valid "
        "moves. Completely eliminate any moves that land on "
        "X-squares (diagonal to an empty corner) or C-squares "
        "(adjacent to an empty corner) from your consideration pool, "
        "unless you have absolutely no other legal moves.\n"
        "3. Exploit the Edge: Among the remaining safe moves, can "
        "you play on an edge square that safely anchors to an "
        "already established stable piece, building an unflippable "
        "wall? If yes, play it.\n"
        "4. Minimize Opponent's Mobility: Among the safe moves, "
        "evaluate how many legal moves your opponent will have "
        "after you play. Choose the move that leaves the opponent "
        "with the fewest possible legal moves on their next turn.\n"
        "5. The Quiet Move: If mobility calculation is complex, "
        "choose the safest 'quiet move.' Select a move that flips "
        "the fewest number of discs, preferably flipping internal "
        "discs rather than discs on the outer perimeter facing the "
        "empty board.\n\n"
        "Action Required: Before making your move, output a brief "
        "step-by-step evaluation (e.g., 'Step 1: No corners "
        "available. Step 2: Eliminating moves at G2 and B7 because "
        "they are dangerous C/X squares. Step 3: No safe edge plays. "
        "Step 4: Move at D3 restricts opponent's mobility "
        "significantly, only flipping one internal disc. I choose "
        "D3.'), and then output your final chosen coordinate."
    ),
}


_BREAKTHROUGH_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "Breakthrough-v0",
    "description": (
        "6-step Breakthrough decision checklist (Immediate Win -> "
        "Immediate Threat Prevention -> Advance the Runner -> Target "
        "Opponent's Breakthrough -> Calculated Captures & Sacrifices "
        "-> Maintain the Phalanx). Tests whether the LLM can reason "
        "about runner/race dynamics and mandatory-capture mechanics, "
        "plus diagonal pawn-chain (phalanx) protection."
    ),
    "pattern_text": (
        "When it is your turn, evaluate your pieces and the board "
        "state by strictly moving down this prioritized checklist. "
        "Execute the top-most applicable logic:\n\n"
        "1. Immediate Win (Touchdown): Can any of your pieces step "
        "onto the opponent's home row (the back rank) on this turn? "
        "If yes, make that move to immediately win the game.\n"
        "2. Immediate Threat Prevention (Crucial): Is there an enemy "
        "piece exactly one step away from your home row? If yes, you "
        "MUST capture it if possible, or move a piece to block its "
        "potential diagonal winning paths.\n"
        "3. Advance the Runner: Do you already have a piece that "
        "has safely bypassed all enemy pieces in its path (an "
        "unstoppable runner)? If yes, advance it forward. If both "
        "players have runners, count the steps: only advance if your "
        "runner will reach the end before the opponent's.\n"
        "4. Target Opponent's Breakthrough: Is an enemy piece "
        "threatening to break through your defensive lines (even if "
        "it's not on the final row yet)? Prioritize capturing it or "
        "maneuvering your pieces to intercept its diagonal capture "
        "zones.\n"
        "5. Calculated Captures & Sacrifices: Can you capture an "
        "opponent's piece to break their defensive chain? "
        "Alternatively, can you deliberately sacrifice a piece to "
        "pull a key enemy defender away, clearing a path for your own "
        "piece to become a runner?\n"
        "6. Maintain the Phalanx (Safe Advance): If no immediate "
        "tactical moves are available, advance a piece safely. "
        "Prioritize moving a piece so that it lands on a square "
        "diagonally protected by another friendly piece behind it. "
        "Avoid pushing isolated pieces forward where they can be "
        "captured for free.\n\n"
        "Action Required: Before making your move, output a brief "
        "step-by-step evaluation (e.g., 'Step 1: No. Step 2: No "
        "immediate enemy threat on the back rank. Step 3: No clear "
        "runners yet. Step 4: Opponent's piece at C5 is dangerous. "
        "Step 5: I can safely capture it with my piece at D4. I "
        "will capture.'), and then output your final chosen move "
        "coordinates."
    ),
}


_KUHNPOKER_REASONING_PATTERN: ReasoningPatternItem = {
    "type": "reasoning_pattern",
    "env_id": "KuhnPoker-v0",
    "description": (
        "3-card GTO decision template for Kuhn Poker. Each card (King "
        "value-bet/trap, Jack bluff/fold, Queen bluff-catch) has its "
        "own action policy parameterized by the betting history and "
        "position. Tests whether the LLM can apply mixed-strategy "
        "(randomized) play for a hidden-info game, plus step-by-step "
        "GTO reasoning before committing to an action."
    ),
    "pattern_text": (
        "When it is your turn, evaluate your decision based strictly "
        "on the Nash Equilibrium (GTO) rules for Kuhn Poker. Execute "
        "the logic corresponding to your card and the betting "
        "history:\n\n"
        "1. If you hold the KING (The Nuts):\n"
        "   - If you act first (P1): You must mix your strategy. "
        "Choose to Bet (~33% of the time for value) or Check (~67% of "
        "the time to trap your opponent into bluffing).\n"
        "   - If you act second (P2) and P1 checked: Always Bet (for "
        "value).\n"
        "   - If you face a Bet: Always Call (you have the unbeatable "
        "card).\n\n"
        "2. If you hold the JACK (The Worst Card):\n"
        "   - If you act first (P1): You must mix your strategy. "
        "Choose to Bet (~33% of the time as a bluff) or Check (~67% "
        "of the time giving up).\n"
        "   - If you act second (P2) and P1 checked: Choose to Bet "
        "(~33% of the time as a bluff) or Check (~67% giving up).\n"
        "   - If you face a Bet: Always Fold (you cannot beat any "
        "card).\n\n"
        "3. If you hold the QUEEN (The Bluff-Catcher):\n"
        "   - If you act first (P1): Always Check. Never bet "
        "initially (a King will call, a Jack will fold, making "
        "betting useless).\n"
        "   - If you act second (P2) and P1 checked: Always Check "
        "(take the free showdown).\n"
        "   - If you face a Bet (as P1 or P2): You are facing either "
        "a King (value) or a Jack (bluff). You must mix your strategy "
        "to avoid being exploited. Choose to Call (~33% to 50% of the "
        "time to catch bluffs) or Fold (the rest of the time).\n\n"
        "Action Required: Before making your move, output a brief "
        "step-by-step evaluation of your GTO position (e.g., 'Step 1: "
        "I hold a Queen. Step 2: I am P2 and P1 just bet. Step 3: As "
        "a bluff-catcher facing a bet, I need to mix calling and "
        "folding. I will decide to Call this time.'), and then output "
        "your final chosen action (Bet, Check, Call, or Fold)."
    ),
}


REASONING_PATTERN_ITEMS: Dict[str, List[ReasoningPatternItem]] = {
    "TicTacToe-v0": [_TICTACTOE_REASONING_PATTERN],
    "ConnectFour-v0": [_CONNECTFOUR_REASONING_PATTERN],
    "Nim-v0": [_NIM_REASONING_PATTERN],
    "UltimateTicTacToe-v0": [_UTTT_REASONING_PATTERN],
    "Othello-v0": [_OTHELLO_REASONING_PATTERN],
    "Breakthrough-v0": [_BREAKTHROUGH_REASONING_PATTERN],
    "KuhnPoker-v0": [_KUHNPOKER_REASONING_PATTERN],
    "LiarsDice-v0": [_LIARSDICE_REASONING_PATTERN],
}


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def _format_trajectory_body(item: TrajectoryItem) -> str:
    """Render the trajectory games (no template wrapper) — just the
    past-game narratives and lessons."""
    blocks = []
    for i, g in enumerate(item["games"], 1):
        marker = "WIN" if g["outcome"] == "WIN" else "LOSS"
        blocks.append(
            f"--- Game {i} ({marker}, vs {g['opponent']}) ---\n"
            f"{g['narrative']}\n"
            f"Lesson: {g['lesson']}"
        )
    return "\n\n".join(blocks)


# Template framing the Memory Item as standing experience to apply.
# Returned text is meant to be the `### Language Policy` body (i.e.
# fed in as `strategy_prompt` to `build_agent_prompt`).
_POLICY_TEMPLATE = (
    "You are a {env_id} player. Your goal is to win the game.\n\n"
    "Before each move, consult the past-game experience below and apply "
    "the lessons learned. Treat the win/loss examples as concrete "
    "evidence of what to do (and what to avoid) in similar positions.\n\n"
    "Past experience:\n{trajectory}"
)


def render_into_policy(env_id: str, item) -> str:
    """
    Render a Memory Item (Trajectory / Insight / Reasoning Pattern)
    into a complete `### Language Policy` body.

    The returned string is meant to be passed as the `strategy_prompt`
    argument to `build_agent_prompt` — i.e. the Memory Item is baked
    into Language Policy, NOT injected as a separate prompt section.

    - Trajectory: wraps the games in the `_POLICY_TEMPLATE` (which
      frames them as "past-game experience to apply").
    - Insight: returns the free-form insight text as-is. Insight items
      are already complete policy text (with role + objective + rules),
      so re-templating them would just add boilerplate and contradict
      their own framing (e.g. "consult the past-game experience" is
      wrong for a rule-based Insight).
    - Reasoning Pattern: returns the checklist text as-is, same
      rationale as Insight — it's already framed as standing rules.
    """
    if item["type"] == "trajectory":
        trajectory = _format_trajectory_body(item)
        return _POLICY_TEMPLATE.format(env_id=env_id, trajectory=trajectory)
    if item["type"] == "insight":
        return item["insight_text"]
    if item["type"] == "reasoning_pattern":
        return item["pattern_text"]
    raise ValueError(f"Unsupported memory item type: {item['type']!r}")


def render_trajectory_section(item: TrajectoryItem) -> str:
    """
    Render a TrajectoryItem into a "### Memory" section text.

    Kept for backwards compatibility / future use; the main path
    (used by `build_policy` and the Bench runner) is
    `render_into_policy` which bakes the trajectory into Language
    Policy instead.
    """
    blocks = [
        "### Memory (Trajectory)",
        (
            f"These are past games of '{item['env_id']}'. Study them "
            f"carefully before choosing each move."
        ),
    ]
    for i, g in enumerate(item["games"], 1):
        marker = "WIN" if g["outcome"] == "WIN" else "LOSS"
        blocks.append(
            f"\n--- Game {i} ({marker}, vs {g['opponent']}) ---\n"
            f"{g['narrative']}\n"
            f"Lesson: {g['lesson']}"
        )
    return "\n".join(blocks)


def get_item(env_id: str, item_type: MemoryItemType = "trajectory"):
    """
    Return the first available item of the given type for the given env,
    or None if the env/type is not registered.
    """
    if item_type == "trajectory":
        registry = TRAJECTORY_ITEMS
    elif item_type == "insight":
        registry = INSIGHT_ITEMS
    elif item_type == "reasoning_pattern":
        registry = REASONING_PATTERN_ITEMS
    else:
        return None
    items = registry.get(env_id)
    if not items:
        return None
    return items[0]


def render_memory(env_id: str, item_type: MemoryItemType = "trajectory") -> Optional[str]:
    """
    One-shot helper: look up the item for `env_id` and render it as a
    full `### Language Policy` body (i.e. with the policy template
    applied). Returns None if no item is registered.
    """
    item = get_item(env_id, item_type)
    if item is None:
        return None
    return render_into_policy(env_id, item)
