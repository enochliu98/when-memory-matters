"""
Play one game between two LLM-driven agents, with full per-call trace.

Each side of the matchup can have its own LLM client AND its own
`include_thinking` flag, so the Bench can match e.g. `thinking` mode
against `non_thinking` mode in a single game.
"""

from datetime import datetime
from math import isfinite
from typing import Callable, Dict, List

from langpsro.envs.textarena_env import TextArenaEnv, parse_action
from langpsro.utils import build_agent_prompt


def make_traced_agent(
    strategy_prompt: str,
    env_id: str,
    llm_client,
    role: str,
    trace: List[Dict],
    include_thinking: bool = False,
) -> Callable[[str], str]:
    """
    Build an agent that, on every observation, builds the full prompt,
    calls the LLM, logs the raw response + parsed action, and returns
    the parsed action.

    The prompt is constructed via `build_agent_prompt(...)` so all
    call sites (this Bench runner, StrategyEvaluator, BRS) share the
    same section layout. The `strategy_prompt` argument should already
    contain any Memory Item content (use `build_policy(env_id, ...)`
    to compose it).
    """
    def agent(observation: str) -> str:
        full_prompt = build_agent_prompt(
            strategy_prompt=strategy_prompt,
            observation=observation,
            env_id=env_id,
            include_thinking=include_thinking,
            output_format_style=("thinking_aware" if include_thinking else "simple"),
        )
        raw = llm_client(full_prompt)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Model returned no action text.")
        action = parse_action(raw, env_id, observation)

        trace.append({
            "role": role,
            "prompt": full_prompt,
            "raw_response": raw,
            "parsed_action": action,
            "timestamp": datetime.now().isoformat(),
        })
        return action

    return agent


def play_one_matchup(
    env_id: str,
    p0_prompt: str,
    p1_prompt: str,
    p0_client,
    p1_client,
    *,
    p0_include_thinking: bool = False,
    p1_include_thinking: bool = False,
) -> Dict:
    """
    Play one full game between two independent LLM agents and return
    the full game data including the per-step LLM trace.

    Both sides can use a different LLM client and a different
    `include_thinking` setting. The two strategy_prompts (p0, p1) can
    be the same or different — the Bench passes the same Memory-Item-
    embedded policy to both sides for mode-vs-mode matches.

    Returns a dict with keys: env_id, p0_prompt, p1_prompt, rewards,
    winner, game_info, trajectory, llm_calls.
    """
    trace: List[Dict] = []

    p0_agent = make_traced_agent(
        strategy_prompt=p0_prompt,
        env_id=env_id,
        llm_client=p0_client,
        role="p0",
        trace=trace,
        include_thinking=p0_include_thinking,
    )
    p1_agent = make_traced_agent(
        strategy_prompt=p1_prompt,
        env_id=env_id,
        llm_client=p1_client,
        role="p1",
        trace=trace,
        include_thinking=p1_include_thinking,
    )

    env = TextArenaEnv(env_id=env_id, num_players=2)
    result = env.play_game(agents=[p0_agent, p1_agent])
    if not result.done:
        raise RuntimeError(f"{env_id} reached the step limit before a terminal outcome.")
    if len(result.rewards) != 2 or not all(isfinite(reward) for reward in result.rewards):
        raise ValueError(f"{env_id} did not return two finite terminal rewards.")

    return {
        "env_id": env_id,
        "p0_prompt": p0_prompt,
        "p1_prompt": p1_prompt,
        "p0_include_thinking": p0_include_thinking,
        "p1_include_thinking": p1_include_thinking,
        "rewards": result.rewards,
        "winner": (
            "p0" if result.rewards[0] > result.rewards[1]
            else "p1" if result.rewards[1] > result.rewards[0]
            else "draw"
        ),
        "game_info": result.game_info,
        "trajectory": result.trajectory,
        "llm_calls": trace,
    }


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------
def play_one_game(
    env_id: str,
    strategy_prompt: str,
    baseline_prompt: str,
    llm_client,
    *,
    include_thinking: bool = False,
) -> Dict:
    """
    Deprecated single-side vs fixed-baseline wrapper.

    Kept so `test_llm_vs_baseline.py` (and any external code) keeps
    working. Internally it just calls `play_one_matchup` with the same
    LLM client and same `include_thinking` flag on both sides.

    The `winner` field is renamed in the result: "p0" / "p1" / "draw"
    (the new convention), with `strategy` mapped from "p0" and
    `baseline` mapped from "p1" for backward compatibility.
    """
    raw = play_one_matchup(
        env_id=env_id,
        p0_prompt=strategy_prompt,
        p1_prompt=baseline_prompt,
        p0_client=llm_client,
        p1_client=llm_client,
        p0_include_thinking=include_thinking,
        p1_include_thinking=include_thinking,
    )
    # Re-map winner label for the old API
    w = raw["winner"]
    raw["winner"] = (
        "strategy" if w == "p0"
        else "baseline" if w == "p1"
        else "draw"
    )
    raw["strategy_prompt"] = raw.pop("p0_prompt")
    raw["baseline_prompt"] = raw.pop("p1_prompt")
    raw["include_thinking"] = raw.pop("p0_include_thinking")
    raw.pop("p1_include_thinking")
    return raw
