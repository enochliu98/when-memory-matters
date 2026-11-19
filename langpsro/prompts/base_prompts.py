"""
Prompt Templates for LangPSRO

Contains all prompt templates for strategy representation,
BRS optimization, and clustering.
"""

# Base prompt for initial strategy
BASE_PROMPT = """You are a strategic game player.

Your task: Play the game to maximize your chances of winning.

Guidelines:
- Analyze the current game state carefully
- Consider possible moves and their consequences
- Prioritize winning moves over defensive moves when safe

Remember: Your goal is to WIN."""


# BRS (Best Response Solver) prompt - similar to Reflexion
BRS_PROMPT = """You are a strategy optimizer for a zero-sum game.

Your task: Analyze game play trajectories and produce a more robust version of
the current strategy.

--- STRATEGY TO OPTIMIZE ---
{current_strategy}
--- END STRATEGY ---

--- GAME PLAY TRAJECTORIES ---
{rollout_trajectory}
--- END TRAJECTORIES ---

Instructions (follow this two-step workflow exactly):

Step 1 - Diagnose.
  Read every trajectory carefully. Each step is annotated with `role`:
  - `role: brs` marks moves made by the strategy we are optimizing.
  - `role: opponent` marks moves made by the opposing strategy.
  Attribute outcomes to the moves that actually caused them (wins/losses).
  Then abstract the per-game observations into GENERAL lessons - patterns
  that recurred across multiple games. Group them as:
    * Common failure modes (mistakes the BRS strategy kept making).
    * Common success patterns (decisions the BRS strategy got right, or
      opponent patterns the BRS strategy exploited well).
  Only include lessons that are supported by the trajectories. Do not
  invent issues that the trajectories do not show. Skip generic advice
  like "analyze more" or "be careful".

Step 2 - Optimize.
  Combine the current strategy with the lessons from Step 1 and write an
  improved version. Keep it under 150 words. Only add or sharpen rules
  that directly address the diagnosed failure modes or reinforce the
  success patterns. Do not pad with generic safety nets.

Output format (use these two sections exactly, with the bracketed
headings on their own lines so they can be parsed automatically):

[LESSONS LEARNED]
- Failure mode 1: <one-line description, citing the kind of game state>
- Failure mode 2: ... (omit if none)
- Success pattern 1: <one-line description>
- Success pattern 2: ... (omit if none)

[IMPROVED STRATEGY]
The full improved strategy text, under 150 words. Do not wrap in a
code block. Do not include the bracketed heading itself in the body.
"""


# Clustering prompt for scheme 4
CLUSTER_PROMPT = """You are a strategy pool manager for a zero-sum game environment.

Your task: Analyze and cluster strategies based on their semantic similarity and strategic behavior.

--- STRATEGY POOL ---
{strategy_pool}
--- END STRATEGY POOL ---

Analyze each strategy and group them into clusters based on:
1. Semantic similarity (similar prompts lead to similar behavior)
2. Strategic goals (offensive, defensive, balanced, etc.)
3. Tactical approaches (aggressive, conservative, adaptive, etc.)

--- OUTPUT FORMAT ---

## Clusters

```json
{{
  "clusters": [
    {{
      "id": 0,
      "strategies": ["strategy_name_1", "strategy_name_2"],
      "theme": "brief cluster theme description",
      "reason_for_merge": "why these strategies belong together"
    }},
    ...
  ],
  "representatives": [
    {{
      "cluster_id": 0,
      "strategy_name": "strategy_name_1",
      "strategy_text": "...",
      "reason_for_selection": "why this was chosen as representative"
    }},
    ...
  ],
  "reduced_pool": ["strategy_name_1", "strategy_name_3", ...]
}}
```

Keep clusters to 3-5 total. Each cluster should have a clear strategic theme."""


# Game simulation prompt (if needed for debugging/analysis)
GAME_SIMULATION_PROMPT = """You are a game simulator for a zero-sum game.

Game: {game_description}

Playing the following strategies against each other:
- Strategy A: {strategy_A}
- Strategy B: {strategy_B}

Simulate the game step by step.

Output format:
## Game Play
[Step-by-step simulation]

## Result
- Winner: A / B / Draw
- Score: [if applicable]
- Key factors that determined the outcome: [brief explanation]
"""


# Meta-strategy analysis prompt
META_STRATEGY_PROMPT = """You are a game theory analyst.

Given a strategy pool and their performance against each other:

--- STRATEGY POOL ---
{strategy_pool}
--- END POOL ---

--- META-PAYOFF MATRIX ---
{meta_payoff_matrix}
--- END MATRIX ---

Analyze the meta-strategy distribution that would be optimal.

Consider:
1. Which strategies are most robust?
2. Which strategies exploit specific weaknesses?
3. What is the Nash equilibrium mix?

Provide your analysis and recommend a meta-strategy distribution."""