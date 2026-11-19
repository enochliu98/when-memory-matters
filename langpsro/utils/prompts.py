"""
Shared prompt construction for LangPSRO agents.

All game-playing agents (StrategyEvaluator, BRS rollout, the Bench runner,
test_llm_vs_baseline) build the same multi-section prompt. This helper
centralizes the layout so adding new optional sections is a one-line
change.

Default sections (4, backward-compatible with current call sites):

    1. ### Language Policy
    2. ### Local Observation
    3. ### Instruction
    4. ### Output Format

Optional section (added between Instruction and Output Format):

    * `### Thinking` — only when `include_thinking=True`

Note: Memory Items (Trajectory / Insight / Reasoning Pattern) are NOT
injected as a separate prompt section. They are baked into the
`### Language Policy` by `build_policy(env_id, memory_item)` (see
`Bench/memory_items.py`). This keeps the prompt 4-section for every
call site, with the standing strategy carrying whatever memory item
the caller wants to test.
"""

from ..envs.textarena_env import get_action_format


# Exact wording of the optional Thinking section. Kept here so the four
# call sites stay byte-identical when include_thinking=True.
THINKING_SECTION = (
    "### Thinking\n"
    "First, reason step by step about the current board state, the legal "
    "moves, and which move best serves your policy. "
    "Write your reasoning here in plain prose."
)


def build_agent_prompt(
    strategy_prompt: str,
    observation: str,
    env_id: str,
    include_thinking: bool = False,
    output_format_style: str = "thinking_aware",
) -> str:
    """
    Build the agent prompt for one LLM call.

    Sections appear in this order:
        1. ### Language Policy
        2. ### Local Observation
        3. ### Instruction
        4. ### Thinking              (only if include_thinking)
        5. ### Output Format         (always last)

    Args:
        strategy_prompt: The FULL `### Language Policy` text — including
            any Memory Item content the caller has chosen to embed.
            Use `build_policy(env_id, memory_item)` for a ready-made
            strategy that combines a fixed template with a registered
            Memory Item.
        observation: TextArena observation the agent currently sees.
        env_id: TextArena env id (e.g. "TicTacToe-v0"). Used to fetch
            the per-env "### Output Format" hint.
        include_thinking: Inject a `### Thinking` section asking the
            model to reason step by step before answering.
        output_format_style: Wording of the Output Format section.
            - "thinking_aware" (default): hints that the model should
              output `Action: [N]` *after* any reasoning prose. Use this
              when `include_thinking=True`.
            - "simple": the original 4-section wording that says
              "Start your reply with 'Action: '" — used by
              StrategyEvaluator and BRS rollout where Thinking is not
              injected.

    Returns:
        The fully assembled prompt string.
    """
    output_format_hint = get_action_format(env_id)["hint"]

    parts = [
        f"### Language Policy\n{strategy_prompt}",
        f"### Local Observation\n{observation}",
        "### Instruction\nBased on your current policy and the observation, "
        "choose the best action to maximize the reward.",
    ]
    if include_thinking:
        parts.append(THINKING_SECTION)

    if output_format_style == "simple":
        parts.append(
            f"### Output Format\n{output_format_hint} "
            f"Start your reply with 'Action: ' followed by the action."
        )
    else:  # "thinking_aware"
        parts.append(
            f"### Output Format\nAfter your reasoning, output the chosen "
            f"action on its own line in this exact format: "
            f"`Action: [your_move]`. For example: `Action: [4]`. "
            f"{output_format_hint}"
        )

    return "\n\n".join(parts)


def build_policy(
    env_id: str,
    memory_item: str = "trajectory",
) -> str:
    """
    Build the `### Language Policy` text by combining a fixed template
    with the Memory Item for the given env. The returned string is
    meant to be passed as `strategy_prompt` to `build_agent_prompt`.

    The template frames the Memory Item as "past experience to apply
    during this game". This keeps every LangPSRO strategy a single text
    blob (no separate Memory section), which matches how PSRO treats
    all strategies uniformly.

    Pass `memory_item=""` (or a type that has no registered item for
    `env_id`) to run an ablation cell without any Memory Item — the
    function falls back to a minimal "play the game to win" template.
    """
    # Lazy import to keep this helper usable outside Bench (e.g. if
    # some other tool wants the same template) and to avoid a hard dep.
    from Bench.memory_items import get_item, render_into_policy

    item = get_item(env_id, memory_item) if memory_item else None
    if item is None:
        return (
            f"You are a {env_id} player. Your goal is to win the game. "
            f"Make smart moves based on the current observation."
        )
    return render_into_policy(env_id, item)
