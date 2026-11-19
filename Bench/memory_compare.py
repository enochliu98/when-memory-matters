"""
Memory Item Comparison experiment for the LangPSRO Bench.

This is the *third* part of the bench evaluation, independent of
`runner.py` (thinking-mode comparison) and `llm_compare.py`
(model comparison). This script holds the **env, thinking mode,
LLM provider, and model name all constant**, varying only the
**memory item type** that gets baked into the policy. It answers:
"given a fixed agent, which kind of memory item (Trajectory /
Insight / Reasoning Pattern) actually makes the policy strongest?"

For N memory items, this script plays an N x N upper-triangle
round-robin. Output is written to:
    results/<env_id>/memory_comparison/<ts>/
        - run_config.json
        - summary.csv
        - <env>__<itemA>_vs_<itemB>__side=...__gameNN.json

This path is distinct from runner.py's and llm_compare.py's
so the three experiments do not collide on disk.

Usage:
    # Compare 3 memory items on TicTacToe with the default model
    python -m Bench.memory_compare --game TicTacToe-v0 \
        --items insight reasoning_pattern trajectory \
        --provider volcengine --thinking_mode non_thinking_pgt \
        --num_games 6

    # Custom model + temperature
    python -m Bench.memory_compare --game ConnectFour-v0 \
        --items insight reasoning_pattern \
        --provider volcengine --model deepseek-r1 \
        --thinking_mode thinking --num_games 4 --temperature 0.3
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import combinations
from typing import Dict, List

# Allow `python -m Bench.memory_compare` from project root.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tqdm is optional — fall back to plain prints if not installed.
try:
    from tqdm import tqdm as _tqdm_factory
    def _tqdm(*args, **kwargs):
        return _tqdm_factory(*args, **kwargs)
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def _tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class _Noop:
                def update(self, n=1): pass
                def set_description(self, s): pass
                def close(self): pass
            return _Noop()
        return iterable

from langpsro.utils import make_llm_client, build_policy
from langpsro.utils.llm_clients import resolve_model
from Bench.config import (
    DEFAULT_OUTPUT_DIR, MODE_CONFIG, SUPPORTED_GAMES, balanced_game_count,
    load_group_or_exit,
)
from Bench.memory_items import (
    TRAJECTORY_ITEMS, INSIGHT_ITEMS, REASONING_PATTERN_ITEMS,
)
from Bench.play import play_one_matchup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Combined memory-item registry. The set of supported `--items` values.
_ITEM_REGISTRY = {
    "trajectory": TRAJECTORY_ITEMS,
    "insight": INSIGHT_ITEMS,
    "reasoning_pattern": REASONING_PATTERN_ITEMS,
}
SUPPORTED_ITEMS = list(_ITEM_REGISTRY.keys())

PROVIDERS = {
    "volcengine": "deepseek-v4-flash",
    "deepseek": "deepseek-chat",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare N memory items under a fixed (env, "
                  "provider, model, thinking mode)."
    )
    p.add_argument(
        "--game", default="TicTacToe-v0", choices=SUPPORTED_GAMES,
        help="TextArena env id (only single env supported for now)",
    )
    p.add_argument(
        "--items", nargs="+", required=True,
        choices=SUPPORTED_ITEMS,
        help=(
            "Whitespace-separated list of memory item TYPES to compare. "
            "All must be registered for --game. Example: "
            "--items insight reasoning_pattern trajectory"
        ),
    )
    p.add_argument(
        "--provider", default="volcengine",
        choices=list(PROVIDERS.keys()),
        help="LLM provider (all matchups use the same provider)",
    )
    p.add_argument(
        "--model", default=None,
        help=(
            "Model name to use under --provider. If omitted, uses the "
            "provider's default (volcengine -> deepseek-v4-flash, "
            "deepseek -> deepseek-chat)."
        ),
    )
    p.add_argument(
        "--thinking_mode", default="non_thinking_pgt",
        choices=list(MODE_CONFIG.keys()),
        help="Thinking mode used by BOTH sides of every matchup",
    )
    p.add_argument(
        "--num_games", "--num-games", type=balanced_game_count, default=6,
        help="Number of games per (item_a, item_b) cell, balanced "
             "50/50 first-mover",
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
    p.add_argument("--memory-group", help="JSON file containing one authored Memory Item group")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cell runner (independent of runner.run_cell / llm_compare.run_cell)
# ---------------------------------------------------------------------------
def run_cell(
    env_id: str,
    item_a: str,
    item_b: str,
    provider: str,
    model: str,
    thinking_mode: str,
    num_games: int,
    temperature: float,
    out_dir: str,
    *,
    policies: dict[str, str] | None = None,
) -> Dict:
    """
    Play `num_games` games between two memory-item-embedded
    policies. Both sides use the same provider, model, and
    thinking mode — only the memory item type differs.

    Result row keys mirror the other bench scripts so downstream
    consumers can detect cell schema via `matchup_type`.
    """
    cfg = MODE_CONFIG[thinking_mode]

    # Build the two strategy prompts (one per memory item).
    strategy_prompt_a = policies[item_a] if policies is not None else build_policy(env_id, item_a)
    strategy_prompt_b = policies[item_b] if policies is not None else build_policy(env_id, item_b)

    # One client per side. Both run under the same thinking-mode
    # config so reasoning behavior is identical; only the policy
    # text (memory-item content) differs.
    common_kwargs = dict(
        temperature=temperature,
        thinking_enabled=cfg["thinking_enabled"],
    )
    if model is not None:
        common_kwargs["model"] = model
    client_a = make_llm_client(provider, **common_kwargs)
    client_b = make_llm_client(provider, **common_kwargs)

    n_p0_first = num_games // 2
    n_p1_first = num_games - n_p0_first

    wins_a = wins_b = draws = 0
    total_reward_a = 0.0
    game_lengths: List[int] = []
    t_start = time.time()
    game_idx = 0

    def _snapshot(c):
        u = c.get_usage_summary()["total"]
        return (u["prompt_tokens"], u["completion_tokens"],
                u["reasoning_tokens"], u["call_count"])

    init_a, init_b = _snapshot(client_a), _snapshot(client_b)
    tokens_a = [0, 0, 0, 0]
    tokens_b = [0, 0, 0, 0]

    def _play(swap: bool):
        nonlocal wins_a, wins_b, draws, total_reward_a, game_idx
        if not swap:
            p0_c, p1_c = client_a, client_b
            p0_prompt, p1_prompt = strategy_prompt_a, strategy_prompt_b
        else:
            p0_c, p1_c = client_b, client_a
            p0_prompt, p1_prompt = strategy_prompt_b, strategy_prompt_a
        before_a, before_b = _snapshot(client_a), _snapshot(client_b)
        result = play_one_matchup(
            env_id=env_id,
            p0_prompt=p0_prompt,
            p1_prompt=p1_prompt,
            p0_client=p0_c,
            p1_client=p1_c,
            p0_include_thinking=cfg["include_thinking"],
            p1_include_thinking=cfg["include_thinking"],
        )
        after_a, after_b = _snapshot(client_a), _snapshot(client_b)
        for i in range(4):
            tokens_a[i] += after_a[i] - before_a[i]
            tokens_b[i] += after_b[i] - before_b[i]
        try:
            r0, r1 = result["rewards"][0], result["rewards"][1]
        except (IndexError, TypeError) as e:
            print(f"      [debug] rewards error: rewards={result.get('rewards')!r}, type={type(result.get('rewards'))}")
            print(f"      [debug] winner={result.get('winner')!r}, game_info={result.get('game_info')!r}")
            raise
        a_won = (r0 > r1) if not swap else (r0 < r1)
        b_won = (r0 < r1) if not swap else (r0 > r1)
        a_reward = r0 if not swap else r1
        if a_won:
            wins_a += 1
        elif b_won:
            wins_b += 1
        else:
            draws += 1
        total_reward_a += a_reward
        game_lengths.append(len(result["trajectory"]))

        fname = (
            f"{env_id}__memory_compare__"
            f"p0={item_a}__p1={item_b}__"
            f"side={'p0_first' if not swap else 'p1_first'}__game{game_idx:02d}.json"
        )
        out = dict(result)
        out["provider"] = provider
        out["model"] = model or PROVIDERS.get(provider, "")
        out["thinking_mode"] = thinking_mode
        out["item_p0"] = item_a if not swap else item_b
        out["item_p1"] = item_b if not swap else item_a
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(out, f, indent=2, default=str)
        game_idx += 1

    for _ in range(n_p0_first):
        _play(swap=False)
    for _ in range(n_p1_first):
        _play(swap=True)

    wall_seconds = time.time() - t_start

    final_a, final_b = _snapshot(client_a), _snapshot(client_b)
    for i, k in enumerate(["prompt", "completion", "reasoning", "call"]):
        assert tokens_a[i] == final_a[i] - init_a[i], (k, tokens_a[i], final_a[i] - init_a[i])
        assert tokens_b[i] == final_b[i] - init_b[i], (k, tokens_b[i], final_b[i] - init_b[i])

    return {
        "env_id": env_id,
        "matchup_type": "memory_comparison",
        "provider": provider,
        "model": model or PROVIDERS.get(provider, ""),
        "thinking_mode": thinking_mode,
        "item_p0": item_a,
        "item_p1": item_b,
        "num_games": num_games,
        "n_p0_first": n_p0_first,
        "n_p1_first": n_p1_first,
        "wins_p0": wins_a,
        "wins_p1": wins_b,
        "draws": draws,
        "win_rate_p0": wins_a / num_games if num_games else 0.0,
        "win_rate_p1": wins_b / num_games if num_games else 0.0,
        "draw_rate": draws / num_games if num_games else 0.0,
        "avg_reward_p0": total_reward_a / num_games if num_games else 0.0,
        "avg_game_length": (
            sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
        ),
        f"prompt_tokens_{item_a}": tokens_a[0],
        f"completion_tokens_{item_a}": tokens_a[1],
        f"reasoning_tokens_{item_a}": tokens_a[2],
        f"call_count_{item_a}": tokens_a[3],
        f"prompt_tokens_{item_b}": tokens_b[0],
        f"completion_tokens_{item_b}": tokens_b[1],
        f"reasoning_tokens_{item_b}": tokens_b[2],
        f"call_count_{item_b}": tokens_b[3],
        "total_prompt_tokens": tokens_a[0] + tokens_b[0],
        "total_completion_tokens": tokens_a[1] + tokens_b[1],
        "total_reasoning_tokens": tokens_a[2] + tokens_b[2],
        "total_calls": tokens_a[3] + tokens_b[3],
        "total_wall_seconds": round(wall_seconds, 2),
    }


def main() -> int:
    args = parse_args()

    items = list(args.items)
    if len(items) < 2:
        raise SystemExit(f"--items requires at least 2 names; got {items}")
    if len(items) != len(set(items)):
        raise SystemExit(f"--items must be unique; got {items}")

    group = load_group_or_exit(args.memory_group, args.game)
    missing = [it for it in items if group is None and args.game not in _ITEM_REGISTRY[it]]
    if missing:
        raise SystemExit(
            f"The following items have no registered entry for env '{args.game}': "
            f"{missing}"
        )

    # Output dir: {output_dir}/{env}/memory_comparison/<ts>/
    # Notes:
    #   - We do NOT use the memory_item folder level here because this
    #     experiment spans multiple items by construction.
    args.model = resolve_model(args.provider, args.model)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(
        args.output_dir, args.game, "memory_comparison", ts
    )
    os.makedirs(run_dir, exist_ok=False)

    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({
            **vars(args),
            "memory_group_id": group.group_id if group else None,
            "memory_group_sha256": group.content_sha256 if group else None,
        }, f, indent=2)

    cells = list(combinations(items, 2))
    total_cells = len(cells)

    print("=" * 70)
    print("Bench: Memory Item Comparison")
    print("=" * 70)
    print(f"Game           : {args.game}")
    print(f"Provider       : {args.provider}")
    print(f"Model          : {args.model or PROVIDERS.get(args.provider, '(default)')}")
    print(f"Thinking mode  : {args.thinking_mode}")
    print(f"Items          : {items}")
    print(f"Cells          : {total_cells}  ({cells})")
    print(f"Games/cell     : {args.num_games}  (balanced 50/50 first-mover)")
    print(f"Temperature    : {args.temperature}")
    print(f"Output dir     : {run_dir}")
    print("=" * 70)

    rows: List[Dict] = []
    bar = _tqdm(total=total_cells, desc="memory_comparison",
                disable=not _HAS_TQDM or args.no_tqdm)
    try:
        for item_a, item_b in cells:
            bar.set_description(f"{item_a} vs {item_b}")
            print(f"\n  --- Cell: p0={item_a} vs p1={item_b} ---")
            try:
                row = run_cell(
                    env_id=args.game,
                    item_a=item_a,
                    item_b=item_b,
                    provider=args.provider,
                    model=args.model,
                    thinking_mode=args.thinking_mode,
                    num_games=args.num_games,
                    temperature=args.temperature,
                    out_dir=run_dir,
                    policies=group.policies if group else None,
                )
            except Exception as e:
                print(f"    ! Cell failed: {type(e).__name__}: {e}")
                bar.update(1)
                continue
            rows.append(row)
            print(
                f"    {item_a}:{row['wins_p0']}  {item_b}:{row['wins_p1']}  "
                f"draws={row['draws']}  prompt_tok={row['total_prompt_tokens']}  "
                f"wall={row['total_wall_seconds']}s"
            )
            bar.update(1)
    finally:
        bar.close()

    csv_path = os.path.join(run_dir, "summary.csv")
    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        prefix = [
            "env_id", "matchup_type", "provider", "model", "thinking_mode",
            "item_p0", "item_p1", "num_games",
            "n_p0_first", "n_p1_first",
            "wins_p0", "wins_p1", "draws",
            "win_rate_p0", "win_rate_p1", "draw_rate",
            "avg_reward_p0", "avg_game_length",
            "total_prompt_tokens", "total_completion_tokens",
            "total_reasoning_tokens", "total_calls",
            "total_wall_seconds",
        ]
        fieldnames = [k for k in prefix if k in all_keys] + [
            k for k in all_keys if k not in prefix
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSummary CSV: {csv_path}  ({len(rows)} cells)")

    print(f"\nAll per-game traces under: {run_dir}")
    if len(rows) != total_cells:
        print("Incomplete run: only successful cells are included in the summary.")
    return int(len(rows) != total_cells)


if __name__ == "__main__":
    sys.exit(main())
