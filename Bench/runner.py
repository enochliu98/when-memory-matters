"""
Bench runner: evaluate mode-vs-mode matches in zero-sum games under
three reasoning modes for the same LLM.

Three reasoning modes (compared for the same LLM):

    1. thinking             — DeepSeek API reasoning ENABLED, no in-prompt Thinking
    2. non_thinking_pgt     — API reasoning DISABLED, in-prompt Thinking section
    3. non_thinking         — API reasoning DISABLED, no in-prompt Thinking

Each unordered pair is played once with balanced sides. A transposed
matrix cell shows the same record in the opposite orientation. Diagonal
cells are not played and do not enter the recorded outcomes.

Usage:
    python -m Bench --game TicTacToe-v0 --num_games 6
    python -m Bench --game all --memory_item insight --num_games 4
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tqdm is optional — fall back to plain prints if not installed.
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable=None, *args, **kwargs):
        """tqdm shim: behave like a no-op iterator / print helper."""
        if iterable is None:
            # Used as a manual progress bar; just return 0 and let the
            # caller call .update(1) harmlessly.
            class _NoopBar:
                def update(self, n=1): pass
                def set_description(self, s): pass
                def close(self): pass
            return _NoopBar()
        return iterable


from langpsro.utils import make_llm_client, build_policy
from langpsro.utils.llm_clients import resolve_model
from Bench.config import (
    DEFAULT_OUTPUT_DIR, MODE_CONFIG, PAPER_GAMES, SUPPORTED_GAMES, balanced_game_count,
    load_group_or_exit,
)
from Bench.memory_items import (
    TRAJECTORY_ITEMS, INSIGHT_ITEMS, REASONING_PATTERN_ITEMS,
)
from Bench.play import play_one_matchup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMES = list(SUPPORTED_GAMES)
THINKING_MODES = list(MODE_CONFIG)
MEMORY_ITEMS_SUPPORTED = ["trajectory", "insight", "reasoning_pattern"]

_ITEM_REGISTRY = {
    "trajectory": TRAJECTORY_ITEMS,
    "insight": INSIGHT_ITEMS,
    "reasoning_pattern": REASONING_PATTERN_ITEMS,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark LLM mode-vs-mode matches in zero-sum games."
    )
    p.add_argument(
        "--game", default="TicTacToe-v0", choices=["all", *GAMES],
        help="Game ID, or 'all' for the six paper games",
    )
    p.add_argument(
        "--memory_item", default="trajectory",
        help=f"Memory item type (choices: {', '.join(MEMORY_ITEMS_SUPPORTED)})",
    )
    p.add_argument(
        "--num_games", "--num-games", type=balanced_game_count, default=6,
        help=(
            "Games per played pairing, split equally between sides. "
            "Default: 6 (3 each side)."
        ),
    )
    p.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM sampling temperature (default 0.3)",
    )
    p.add_argument(
        "--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write per-run subdirectories",
    )
    p.add_argument(
        "--no_tqdm", action="store_true",
        help="Disable tqdm progress bars even if installed.",
    )
    p.add_argument(
        "--llm", type=str, default="volcengine",
        choices=["volcengine", "deepseek"],
        help="LLM provider to use (default: volcengine)",
    )
    p.add_argument("--model", help="Explicit model ID served by --llm")
    p.add_argument("--comparison", choices=("reasoning", "baseline", "both"), default="both")
    p.add_argument(
        "--baseline-mode", choices=("matched", "non_thinking_pgt"), default="matched",
        help="Match the policy's reasoning interface, or use the legacy Prompt-guided baseline",
    )
    p.add_argument("--memory-group", help="JSON file containing one authored Memory Item group")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------
def run_cell(
    env_id: str,
    memory_item: str,
    mode_a: str,
    mode_b: str,
    num_games: int,
    temperature: float,
    out_dir: str,
    llm_provider: str = "volcengine",
    *,
    model: Optional[str] = None,
    policy_text: Optional[str] = None,
) -> Dict:
    """
    Play `num_games` games for one off-diagonal cell (mode_a, mode_b)
    with mode_a != mode_b, balancing the first-mover 50/50. Returns
    the aggregated metrics row (and writes per-game trace JSONs).

    Result-row semantics are *strategy-anchored*, not slot-anchored:
    `wins_p0` counts games where strategy `mode_a` won (regardless of
    whether mode_a was physically at p0 or p1 in that particular
    game). Likewise `wins_p1` counts games where `mode_b` won. This
    is what makes the (mode_a, mode_b) and (mode_b, mode_a) rows
    mirror each other without running a second matchup.
    """
    if mode_a == mode_b:
        raise ValueError(f"An interface pairing requires two different modes; got {mode_a!r}.")
    cfg_a = MODE_CONFIG[mode_a]
    cfg_b = MODE_CONFIG[mode_b]

    # Both sides share the SAME memory-item-embedded language policy.
    # The only difference between mode_a and mode_b is the thinking
    # mode (API reasoning toggle + in-prompt `### Thinking` section).
    strategy_prompt = policy_text if policy_text is not None else build_policy(env_id, memory_item)
    strategy_prompt_a = strategy_prompt
    strategy_prompt_b = strategy_prompt

    # Split first-mover 50/50: first half p0 goes first, second half p1.
    n_p0_first = num_games // 2
    n_p1_first = num_games - n_p0_first

    # Per-side clients (one each — reasoning is independently enabled/disabled).
    client_a = make_llm_client(
        llm_provider,
        model=model,
        temperature=temperature,
        thinking_enabled=cfg_a["thinking_enabled"],
    )
    client_b = make_llm_client(
        llm_provider,
        model=model,
        temperature=temperature,
        thinking_enabled=cfg_b["thinking_enabled"],
    )

    wins_p0 = wins_p1 = draws = 0
    total_reward_p0 = 0.0
    game_lengths: List[int] = []
    t_start = time.time()
    game_idx = 0

    # Per-strategy token tally. In a swap_sides=True game, mode_a sits
    # at p1 (client_a) — so the "mode_a" counter for that game reads
    # client_a's usage delta after the game. We capture per-game
    # before/after snapshots from each client to attribute tokens to
    # the strategy that incurred them, not to the physical slot.
    def _snapshot(c):
        u = c.get_usage_summary()["total"]
        return (u["prompt_tokens"], u["completion_tokens"],
                u["reasoning_tokens"], u["call_count"])

    init_a = _snapshot(client_a)
    init_b = _snapshot(client_b)
    tokens_a = [0, 0, 0, 0]   # prompt, completion, reasoning, calls — used by mode_a
    tokens_b = [0, 0, 0, 0]   # ...used by mode_b

    def _play_one_game(swap_sides: bool) -> None:
        """
        Run one game. If swap_sides, treat client_b as the physical p0
        and client_a as p1 — but the *tally* is ALWAYS in
        (mode_a, mode_b) order, regardless of who actually sat at p0.
        That is, `wins_p0` counts games where mode_a's strategy ended
        up winning, even if in this game mode_a was physically p1.
        """
        nonlocal wins_p0, wins_p1, draws, total_reward_p0, game_idx
        if not swap_sides:
            p0_c, p1_c = client_a, client_b
            p0_it, p1_it = cfg_a["include_thinking"], cfg_b["include_thinking"]
        else:
            # mode_b sits at p0, mode_a sits at p1
            p0_c, p1_c = client_b, client_a
            p0_it, p1_it = cfg_b["include_thinking"], cfg_a["include_thinking"]
        # Snapshot before so we can compute this game's tokens per client.
        before_a = _snapshot(client_a)
        before_b = _snapshot(client_b)
        result = play_one_matchup(
            env_id=env_id,
            p0_prompt=strategy_prompt_a,
            p1_prompt=strategy_prompt_b,
            p0_client=p0_c,
            p1_client=p1_c,
            p0_include_thinking=p0_it,
            p1_include_thinking=p1_it,
        )
        after_a = _snapshot(client_a)
        after_b = _snapshot(client_b)
        # Per-client per-game deltas:
        delta_a = [after_a[i] - before_a[i] for i in range(4)]
        delta_b = [after_b[i] - before_b[i] for i in range(4)]
        # Attribute by strategy, not slot:
        if not swap_sides:
            # client_a = mode_a, client_b = mode_b
            for i in range(4):
                tokens_a[i] += delta_a[i]
                tokens_b[i] += delta_b[i]
        else:
            # client_a = mode_a (sits at p1), client_b = mode_b (sits at p0)
            for i in range(4):
                tokens_a[i] += delta_a[i]
                tokens_b[i] += delta_b[i]

        # Map p0/p1 reward to (mode_a, mode_b) semantics:
        #   wins_p0  = mode_a won
        #   wins_p1  = mode_b won
        r0, r1 = result["rewards"][0], result["rewards"][1]
        if not swap_sides:
            # p0 is mode_a, p1 is mode_b
            a_won, b_won = r0 > r1, r0 < r1
            a_reward = r0
        else:
            # p0 is mode_b, p1 is mode_a
            a_won, b_won = r0 < r1, r0 > r1
            a_reward = r1   # reward received by the side holding mode_a
        if a_won:
            wins_p0 += 1
        elif b_won:
            wins_p1 += 1
        else:
            draws += 1
        total_reward_p0 += a_reward
        game_lengths.append(len(result["trajectory"]))

        # Per-game JSON
        fname = (
            f"{env_id}__{memory_item}__"
            f"p0={mode_a}__p1={mode_b}__"
            f"side={'p0_first' if not swap_sides else 'p1_first'}__game{game_idx:02d}.json"
        )
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(result, f, indent=2, default=str)
        game_idx += 1

    # First half: p0 (mode_a) goes first
    for _ in range(n_p0_first):
        _play_one_game(swap_sides=False)
    # Second half: p1 (mode_b) goes first
    for _ in range(n_p1_first):
        _play_one_game(swap_sides=True)

    wall_seconds = time.time() - t_start

    # Sanity: tokens_a + tokens_b should match the deltas from init.
    final_a = _snapshot(client_a)
    final_b = _snapshot(client_b)
    for i, key in enumerate(["prompt_tokens", "completion_tokens",
                            "reasoning_tokens", "call_count"]):
        assert tokens_a[i] == final_a[i] - init_a[i], (
            f"tokens_a[{key}]={tokens_a[i]} != client_a delta "
            f"{final_a[i] - init_a[i]}"
        )
        assert tokens_b[i] == final_b[i] - init_b[i], (
            f"tokens_b[{key}]={tokens_b[i]} != client_b delta "
            f"{final_b[i] - init_b[i]}"
        )

    row = {
        "env_id": env_id,
        "memory_item": memory_item,
        "mode_p0": mode_a,
        "mode_p1": mode_b,
        "num_games": num_games,
        "n_p0_first": n_p0_first,
        "n_p1_first": n_p1_first,
        "wins_p0": wins_p0,
        "wins_p1": wins_p1,
        "draws": draws,
        "win_rate_p0": wins_p0 / num_games if num_games else 0.0,
        "win_rate_p1": wins_p1 / num_games if num_games else 0.0,
        "draw_rate": draws / num_games if num_games else 0.0,
        "avg_reward_p0": total_reward_p0 / num_games if num_games else 0.0,
        "avg_game_length": (
            sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
        ),
        # Per-strategy token usage (NOT summed across A and B).
        f"prompt_tokens_{mode_a}": tokens_a[0],
        f"completion_tokens_{mode_a}": tokens_a[1],
        f"reasoning_tokens_{mode_a}": tokens_a[2],
        f"call_count_{mode_a}": tokens_a[3],
        f"prompt_tokens_{mode_b}": tokens_b[0],
        f"completion_tokens_{mode_b}": tokens_b[1],
        f"reasoning_tokens_{mode_b}": tokens_b[2],
        f"call_count_{mode_b}": tokens_b[3],
        # Sum row (handy for a single "total cost" glance).
        "total_prompt_tokens": tokens_a[0] + tokens_b[0],
        "total_completion_tokens": tokens_a[1] + tokens_b[1],
        "total_reasoning_tokens": tokens_a[2] + tokens_b[2],
        "total_calls": tokens_a[3] + tokens_b[3],
        "total_wall_seconds": round(wall_seconds, 2),
    }
    return row


# ---------------------------------------------------------------------------
# Phase 2: vs-baseline matchup
# ---------------------------------------------------------------------------
def run_baseline_matchup(
    env_id: str,
    memory_item: str,
    thinking_mode: str,
    num_games: int,
    temperature: float,
    out_dir: str,
    llm_provider: str = "volcengine",
    *,
    model: Optional[str] = None,
    policy_text: Optional[str] = None,
    baseline_mode: str = "matched",
) -> Dict:
    """
    Play `num_games` games for one vs-baseline cell (phase 2).

    Match-up details:
      * The policy supplies the selected Memory Item.
      * By default the baseline uses no Memory Item and the same model and
        reasoning interface. The legacy Prompt-guided baseline is opt-in.

    First-mover is split 50/50 between the two sides to neutralize
    any first-move advantage. No mirror is computed; result is
    role-anchored (policy vs baseline), not slot-anchored.

    The returned row has a DIFFERENT schema than `run_cell`'s row:
      * `thinking_mode`: which thinking mode the policy side used
        (one of THINKING_MODES)
      * `wins_policy` / `wins_baseline` / `draws`
      * `win_rate_policy` / `win_rate_baseline` / `draw_rate`
      * `n_policy_first` / `n_baseline_first`
      * `prompt_tokens_policy` / `completion_tokens_policy` /
        `reasoning_tokens_policy` / `call_count_policy`
      * `prompt_tokens_baseline` / `completion_tokens_baseline` /
        `reasoning_tokens_baseline` / `call_count_baseline`
      * `total_*` and `total_wall_seconds`
    Consumers should filter rows on `matchup_type` (set by the
    main loop) to interpret which schema each row uses.
    """
    from langpsro.utils import get_baseline_prompt

    assert thinking_mode in MODE_CONFIG, (
        f"Unknown thinking_mode: {thinking_mode!r}. "
        f"Choose from {list(MODE_CONFIG)}."
    )
    cfg_policy = MODE_CONFIG[thinking_mode]
    baseline_interface = thinking_mode if baseline_mode == "matched" else baseline_mode
    cfg_baseline = MODE_CONFIG[baseline_interface]
    policy_prompt = policy_text if policy_text is not None else build_policy(env_id, memory_item)
    baseline_prompt = (
        build_policy(env_id, "") if baseline_mode == "matched" else get_baseline_prompt(env_id)
    )

    n_policy_first = num_games // 2
    n_baseline_first = num_games - n_policy_first

    client_policy = make_llm_client(
        llm_provider,
        model=model,
        temperature=temperature,
        thinking_enabled=cfg_policy["thinking_enabled"],
    )
    client_baseline = make_llm_client(
        llm_provider,
        model=model,
        temperature=temperature,
        thinking_enabled=cfg_baseline["thinking_enabled"],
    )

    wins_policy = wins_baseline = draws = 0
    game_lengths: List[int] = []
    t_start = time.time()
    game_idx = 0

    def _snapshot(c):
        u = c.get_usage_summary()["total"]
        return (u["prompt_tokens"], u["completion_tokens"],
                u["reasoning_tokens"], u["call_count"])

    init_pol = _snapshot(client_policy)
    init_base = _snapshot(client_baseline)
    tokens_policy = [0, 0, 0, 0]
    tokens_baseline = [0, 0, 0, 0]

    def _play_one_game(swap_sides: bool) -> None:
        """
        Run one game. If `swap_sides`, baseline sits at p0 and policy
        at p1 (so baseline goes first). Tally is always ROLE-anchored
        (wins_policy / wins_baseline), not slot-anchored.
        """
        nonlocal wins_policy, wins_baseline, draws, game_idx
        if not swap_sides:
            # p0 = policy, p1 = baseline
            p0_prompt, p0_client = policy_prompt, client_policy
            p0_it = cfg_policy["include_thinking"]
            p1_prompt, p1_client = baseline_prompt, client_baseline
            p1_it = cfg_baseline["include_thinking"]
        else:
            # p0 = baseline, p1 = policy
            p0_prompt, p0_client = baseline_prompt, client_baseline
            p0_it = cfg_baseline["include_thinking"]
            p1_prompt, p1_client = policy_prompt, client_policy
            p1_it = cfg_policy["include_thinking"]
        before_pol = _snapshot(client_policy)
        before_base = _snapshot(client_baseline)
        result = play_one_matchup(
            env_id=env_id,
            p0_prompt=p0_prompt,
            p1_prompt=p1_prompt,
            p0_client=p0_client,
            p1_client=p1_client,
            p0_include_thinking=p0_it,
            p1_include_thinking=p1_it,
        )
        after_pol = _snapshot(client_policy)
        after_base = _snapshot(client_baseline)
        delta_pol = [after_pol[i] - before_pol[i] for i in range(4)]
        delta_base = [after_base[i] - before_base[i] for i in range(4)]
        for i in range(4):
            tokens_policy[i] += delta_pol[i]
            tokens_baseline[i] += delta_base[i]

        # Role-anchored win attribution.
        r0, r1 = result["rewards"][0], result["rewards"][1]
        if not swap_sides:
            policy_won, baseline_won = r0 > r1, r0 < r1
        else:
            policy_won, baseline_won = r0 < r1, r0 > r1
        if policy_won:
            wins_policy += 1
        elif baseline_won:
            wins_baseline += 1
        else:
            draws += 1
        game_lengths.append(len(result["trajectory"]))

        # Per-game JSON
        fname = (
            f"{env_id}__{memory_item}__"
            f"policy={thinking_mode}__vs_baseline__"
            f"side={'pol_first' if not swap_sides else 'base_first'}__"
            f"game{game_idx:02d}.json"
        )
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(result, f, indent=2, default=str)
        game_idx += 1

    for _ in range(n_policy_first):
        _play_one_game(swap_sides=False)
    for _ in range(n_baseline_first):
        _play_one_game(swap_sides=True)

    wall_seconds = time.time() - t_start

    # Sanity: per-role tokens match observed client deltas.
    final_pol = _snapshot(client_policy)
    final_base = _snapshot(client_baseline)
    for i, key in enumerate(["prompt_tokens", "completion_tokens",
                            "reasoning_tokens", "call_count"]):
        assert tokens_policy[i] == final_pol[i] - init_pol[i], (
            f"tokens_policy[{key}] mismatch"
        )
        assert tokens_baseline[i] == final_base[i] - init_base[i], (
            f"tokens_baseline[{key}] mismatch"
        )

    return {
        "env_id": env_id,
        "memory_item": memory_item,
        "thinking_mode": thinking_mode,
        "baseline_thinking_mode": baseline_interface,
        "num_games": num_games,
        "n_policy_first": n_policy_first,
        "n_baseline_first": n_baseline_first,
        # Role-anchored outcome
        "wins_policy": wins_policy,
        "wins_baseline": wins_baseline,
        "draws": draws,
        "win_rate_policy": wins_policy / num_games if num_games else 0.0,
        "win_rate_baseline": wins_baseline / num_games if num_games else 0.0,
        "draw_rate": draws / num_games if num_games else 0.0,
        "avg_game_length": (
            sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
        ),
        # Per-role token breakdown
        "prompt_tokens_policy": tokens_policy[0],
        "completion_tokens_policy": tokens_policy[1],
        "reasoning_tokens_policy": tokens_policy[2],
        "call_count_policy": tokens_policy[3],
        "prompt_tokens_baseline": tokens_baseline[0],
        "completion_tokens_baseline": tokens_baseline[1],
        "reasoning_tokens_baseline": tokens_baseline[2],
        "call_count_baseline": tokens_baseline[3],
        # Sum of both sides
        "total_prompt_tokens": tokens_policy[0] + tokens_baseline[0],
        "total_completion_tokens": tokens_policy[1] + tokens_baseline[1],
        "total_reasoning_tokens": tokens_policy[2] + tokens_baseline[2],
        "total_calls": tokens_policy[3] + tokens_baseline[3],
        "total_wall_seconds": round(wall_seconds, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    games = PAPER_GAMES if args.game == "all" else (args.game,)
    modes = THINKING_MODES
    memory_item = args.memory_item
    if memory_item not in MEMORY_ITEMS_SUPPORTED:
        raise SystemExit(
            f"--memory_item must be one of {MEMORY_ITEMS_SUPPORTED}; got {memory_item!r}"
        )

    if args.memory_group and args.game == "all":
        raise SystemExit("--memory-group requires a single --game")
    group = load_group_or_exit(args.memory_group, args.game)
    policy_text = group.policies[memory_item] if group else None
    for g in games:
        if not group and g not in _ITEM_REGISTRY.get(memory_item, {}):
            raise SystemExit(
                f"No {memory_item} item registered for env '{g}'."
            )

    args.model = resolve_model(args.llm, args.model)
    pairs = list(_iter_upper_triangle(modes)) if args.comparison != "baseline" else []
    baseline_modes = modes if args.comparison != "reasoning" else []
    incomplete = False
    use_tqdm = _HAS_TQDM and not args.no_tqdm

    # 2-level directory structure under args.output_dir:
    #   {env_id}/{memory_item}/{timestamp}__{provider}/
    # The first level groups by game, the second by memory item, the
    # third (timestamped) is a single run of that (game, memory_item)
    # cell. Different runs of the same cell accumulate as
    # timestamped sub-folders (e.g. for comparing different LLM
    # providers or temperatures). Because all envs in a CLI run
    # share the same memory_item, we use a single timestamped leaf
    # that covers the whole CLI invocation; if you need finer
    # granularity, invoke the CLI once per env.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_leaf = f"{ts}__{args.llm}"
    env_dirs: List[str] = []

    print("=" * 70)
    print("Bench: LLM mode-vs-mode matches in zero-sum games")
    print("=" * 70)
    print(f"Games      : {games}")
    print(f"Memory item: {memory_item}")
    print(f"Model      : {args.model}")
    print(f"Comparison : {args.comparison}")
    print(f"Games/pair : {args.num_games} (balanced sides)")
    print(f"Temperature: {args.temperature}")
    print(f"Output dir : {args.output_dir}/<env_id>/{memory_item}/{run_leaf}")
    print(f"tqdm       : {use_tqdm}")
    print("=" * 70)

    # Per-env accumulation. Each env writes its own summary.csv and
    # plot inside its own subdirectory.
    env_rows_list: List[List[Dict]] = []

    # Outer tqdm: total work = sum over envs of (3 upper-triangle cells).
    # Diagonal cells (mode, mode) are excluded (assumed draw) and the
    # lower triangle (b, a) is the mirror of (a, b).
    total_cells = len(games) * (len(pairs) + len(baseline_modes))
    outer = tqdm(
        total=total_cells,
        desc="overall",
        disable=not use_tqdm,
    )
    try:
        for env_id in games:
            print(f"\n>>> Env: {env_id}")

            # Per-env subdirectory: {output_dir}/{env_id}/{memory_item}/{run_leaf}/
            env_run_dir = os.path.join(
                args.output_dir, env_id, memory_item, run_leaf
            )
            os.makedirs(env_run_dir, exist_ok=False)
            env_dirs.append(env_run_dir)

            # Write run_config.json inside each env's directory so
            # results are self-describing even if the user copies a
            # single subfolder elsewhere.
            with open(os.path.join(env_run_dir, "run_config.json"), "w") as f:
                json.dump({
                    **vars(args), "game": env_id,
                    "memory_group_id": group.group_id if group else None,
                    "memory_group_sha256": group.content_sha256 if group else None,
                }, f, indent=2)

            env_rows: List[Dict] = []
            for mode_a, mode_b in pairs:
                outer.set_description(f"{env_id} | p0={mode_a} vs p1={mode_b}")
                print(f"\n  --- Cell: p0={mode_a} vs p1={mode_b} ---")
                try:
                    row = run_cell(
                        env_id=env_id,
                        memory_item=memory_item,
                        mode_a=mode_a,
                        mode_b=mode_b,
                        num_games=args.num_games,
                        temperature=args.temperature,
                        out_dir=env_run_dir,
                        llm_provider=args.llm,
                        model=args.model,
                        policy_text=policy_text,
                    )
                except Exception as e:
                    print(f"    ! Cell failed: {type(e).__name__}: {e}")
                    incomplete = True
                    outer.update(1)
                    continue
                env_rows.append(row)
                print(
                    f"    wins_p0={row['wins_p0']} wins_p1={row['wins_p1']} "
                    f"draws={row['draws']}  "
                    f"prompt_tok={row['total_prompt_tokens']} "
                    f"reason_tok={row['total_reasoning_tokens']} "
                    f"wall={row['total_wall_seconds']}s"
                )
                outer.update(1)

            if pairs:
                _print_matrix(env_id, modes, env_rows)

            # ----------------------------------------------------------------
            # Phase 2: vs-baseline matchups.
            # ----------------------------------------------------------------
            for mode in baseline_modes:
                outer.set_description(
                    f"{env_id} | policy({mode}) vs baseline"
                )
                print(f"\n  --- Cell: policy={mode} vs baseline ---")
                try:
                    row = run_baseline_matchup(
                        env_id=env_id,
                        memory_item=memory_item,
                        thinking_mode=mode,
                        num_games=args.num_games,
                        temperature=args.temperature,
                        out_dir=env_run_dir,
                        llm_provider=args.llm,
                        model=args.model,
                        policy_text=policy_text,
                        baseline_mode=args.baseline_mode,
                    )
                except Exception as e:
                    print(f"    ! Cell failed: {type(e).__name__}: {e}")
                    incomplete = True
                    outer.update(1)
                    continue
                row["matchup_type"] = "vs_baseline"
                env_rows.append(row)
                outer.update(1)
                print(
                    f"    wins_policy={row['wins_policy']} "
                    f"wins_baseline={row['wins_baseline']} "
                    f"draws={row['draws']}  "
                    f"prompt_tok(policy/baseline)={row['prompt_tokens_policy']}/"
                    f"{row['prompt_tokens_baseline']} "
                    f"reason_tok(policy/baseline)={row['reasoning_tokens_policy']}/"
                    f"{row['reasoning_tokens_baseline']} "
                    f"wall={row['total_wall_seconds']}s"
                )

            # Persist this env's rows for the post-loop CSV+plot step.
            env_rows_list.append(env_rows)
    finally:
        outer.close()

    # Per-env summary.csv — written into each env's own subdirectory
    # ({args.output_dir}/{env_id}/{memory_item}/{ts}__{llm}/summary.csv).
    # Each row is tagged with a `matchup_type` column:
    #   - "matrix" for cells in the 3x3 thinking-mode matrix
    #   - "vs_baseline" for cells comparing a memory-item policy
    #     against the env's bare baseline (no memory item).
    # Consumers can filter by `matchup_type` to analyze one or the
    # other.
    shared_meta = [
        "env_id", "memory_item", "matchup_type", "num_games",
    ]
    matrix_schema = [
        # Slot-anchored (mode_p0 vs mode_p1)
        "mode_p0", "mode_p1",
        "n_p0_first", "n_p1_first",
        "wins_p0", "wins_p1",
        "win_rate_p0", "win_rate_p1",
        "avg_reward_p0",
    ]
    baseline_schema = [
        # Role-anchored (policy vs baseline)
        "thinking_mode",
        "n_policy_first", "n_baseline_first",
        "wins_policy", "wins_baseline",
        "win_rate_policy", "win_rate_baseline",
    ]
    shared_outcome = ["draws", "draw_rate", "avg_game_length"]
    shared_totals = [
        "total_prompt_tokens", "total_completion_tokens",
        "total_reasoning_tokens", "total_calls",
        "total_wall_seconds",
    ]
    fixed_prefix = (
        shared_meta
        + matrix_schema
        + baseline_schema
        + shared_outcome
        + shared_totals
    )

    for env_id, env_dir, env_rows in zip(games, env_dirs, env_rows_list):
        if not env_rows:
            print(f"\n[skip] {env_dir}: no rows")
            continue

        # Default-fill `matchup_type` for any row not explicitly tagged.
        for r in env_rows:
            r.setdefault("matchup_type", "matrix")

        all_keys = set()
        for r in env_rows:
            all_keys.update(r)
        tail_keys = sorted(all_keys - set(fixed_prefix))
        fieldnames = [k for k in fixed_prefix if k in all_keys] + tail_keys

        csv_path = os.path.join(env_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in env_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        n_matrix = sum(1 for r in env_rows if r.get("matchup_type") == "matrix")
        n_baseline = sum(1 for r in env_rows if r.get("matchup_type") == "vs_baseline")
        print(f"\nSummary CSV [{env_id}]: {csv_path}  "
              f"({n_matrix} matrix cells + {n_baseline} vs-baseline cells, "
              f"{len(fieldnames)} columns)")

        # Per-env plot
        if n_matrix:
            try:
                from Bench.plot import plot_summary
                plot_path = plot_summary(env_dir)
                print(f"Summary plot [{env_id}]: {plot_path}")
            except (SystemExit, ImportError) as e:
                print(f"Plot skipped [{env_id}]: {e}")
            except Exception as e:
                print(f"Plot skipped [{env_id}]: {type(e).__name__}: {e}")

    print(f"\nAll per-game traces under:")
    for env_dir in env_dirs:
        print(f"  {env_dir}")
    if incomplete:
        print("Incomplete run: only successful cells are included in the summaries.")
    return int(incomplete)


def _print_matrix(env_id: str, modes: List[str], rows: List[Dict]) -> None:
    print(f"\n  Win-rate matrix for {env_id} (row wins / draws / column wins)")
    print("  row/column   " + "  ".join(f"{mode:>20s}" for mode in modes))
    for a in modes:
        row_values = []
        for b in modes:
            forward = _get_cell(rows, env_id, a, b)
            reverse = _get_cell(rows, env_id, b, a)
            if forward:
                values = (forward["win_rate_p0"], forward["draw_rate"], forward["win_rate_p1"])
            elif reverse:
                values = (reverse["win_rate_p1"], reverse["draw_rate"], reverse["win_rate_p0"])
            else:
                row_values.append(f"{'--':>20s}")
                continue
            row_values.append("/".join(f"{value:>5.1%}" for value in values).rjust(20))
        print(f"  {a:12s}" + "  ".join(row_values))


def _iter_upper_triangle(modes: List[str]):
    """Yield the 3 (mode_a, mode_b) pairs in the strict upper triangle
    of the modes list, in order: (modes[0], modes[1]),
    (modes[0], modes[2]), (modes[1], modes[2]). For 3 modes this
    is 3 cells; for N modes it's N*(N-1)/2.
    """
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            yield a, b


def _get_cell(rows: List[Dict], env_id: str, mode_a: str, mode_b: str) -> Optional[Dict]:
    """Look up the row for a specific (env, mode_p0, mode_p1)."""
    for r in rows:
        if (r["env_id"] == env_id
                and r["mode_p0"] == mode_a
                and r["mode_p1"] == mode_b):
            return r
    return None


if __name__ == "__main__":
    sys.exit(main())
