"""
Utils package for LangPSRO
"""
from .llm_clients import (
    DeepSeekChatClient,
    VolcengineChatClient,
    create_deepseek_client,
    create_volcengine_client,
)
from .prompts import build_agent_prompt, build_policy
from .evaluator import (
    StrategyEvaluator,
    ExperimentLogger,
    EvaluationResult,
    RoundResult,
    BASELINE_PROMPT,
    BASELINE_PROMPTS,
    get_baseline_prompt,
)


# LLM factory — pick the provider via a single name. New code should
# call `make_llm_client("volcengine")` (or "deepseek") instead of
# importing the concrete classes directly, so swapping providers is
# a one-line change.
def make_llm_client(provider: str = "volcengine", **kwargs):
    """
    Build an LLM client by name.

    Args:
        provider: "volcengine" (default) or "deepseek".
        **kwargs: Forwarded to the client constructor (e.g.
            temperature, thinking_enabled, max_tokens).

    Returns:
        Either a `VolcengineChatClient` or `DeepSeekChatClient`.
    """
    if provider == "volcengine":
        return VolcengineChatClient(**kwargs)
    if provider == "deepseek":
        return DeepSeekChatClient(**kwargs)
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Choose from: 'volcengine', 'deepseek'."
    )


__all__ = [
    # LLM clients
    "DeepSeekChatClient",
    "VolcengineChatClient",
    "create_deepseek_client",
    "create_volcengine_client",
    "make_llm_client",
    # Prompts
    "build_agent_prompt",
    "build_policy",
    # Evaluator
    "StrategyEvaluator",
    "ExperimentLogger",
    "EvaluationResult",
    "RoundResult",
    "BASELINE_PROMPT",
    "BASELINE_PROMPTS",
    "get_baseline_prompt",
]
