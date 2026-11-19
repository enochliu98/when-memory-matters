"""
Model Comparison experiment for the LangPSRO Bench.

This is the *second* part of the bench evaluation, independent of
`runner.py` (which compares thinking modes for one LLM). This script
holds the **thinking mode constant** at `non_thinking_pgt` and the
**LLM provider constant**, varying only the **model name** instead, to
answer: "given the same provider + the same prompt, which model
produces the strongest policy?"

For N model names, this script plays an N x N upper-triangle
round-robin (same as the 3x3 thinking-mode matrix in runner.py).
Output is written to:
    results/<env_id>/<memory_item>/model_comparison/<ts>/
        - run_config.json
        - summary.csv
        - <env>__<item>__<modelA>_vs_<modelB>__side=...__gameNN.json

This path is distinct from runner.py's
    results/<env>/<item>/<ts>__<llm>/
and from any future llm_provider_comparison, so the experiments do
not collide on disk.

Usage:
    # 3 models on the same provider
    python -m Bench.llm_compare --game TicTacToe-v0 --memory_item insight \
        --provider volcengine \
        --models deepseek-v4-flash deepseek-r1 qwen3-max \
        --num_games 6

    # With a custom temperature
    python -m Bench.llm_compare --game ConnectFour-v0 --memory_item insight \
        --provider volcengine \
        --models deepseek-v4-flash qwen3-max \
        --temperature 0.3
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
from urllib.parse import quote

# Allow `python -m Bench.llm_compare` from project root.
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
from Bench.config import DEFAULT_OUTPUT_DIR, SUPPORTED_GAMES, balanced_game_count, load_group_or_exit
from Bench.memory_items import (
    TRAJECTORY_ITEMS, INSIGHT_ITEMS, REASONING_PATTERN_ITEMS,
)
from Bench.play import play_one_matchup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_THINKING_MODE = "non_thinking_pgt"   # all model cells use this config

# Provider registry: maps CLI --provider string to (default model name,
# env-var key for the provider's API key). To add a new provider,
# extend this dict AND add a `<Provider>ChatClient` in
# langpsro/utils/llm_clients.py + a `make_llm_client` branch in
# langpsro/utils/__init__.py.
PROVIDERS: Dict[str, str] = {
    "volcengine": "deepseek-v4-flash",   # default model
    "deepseek": "deepseek-chat",
}


# Combined registry used to check whether the requested
# (env_id, memory_item) combination is supported.
_ITEM_REGISTRY = {
    "trajectory": TRAJECTORY_ITEMS,
    "insight": INSIGHT_ITEMS,
    "reasoning_pattern": REASONING_PATTERN_ITEMS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare N models from the same LLM provider under "
                  "a fixed thinking mode."
    )
    p.add_argument(
        "--game", default="TicTacToe-v0", choices=SUPPORTED_GAMES,
        help="TextArena env id (only single env supported for now)",
    )
    p.add_argument(
        "--memory_item", default="insight",
        choices=["trajectory", "insight", "reasoning_pattern"],
        help="Memory item type",
    )
    p.add_argument(
        "--provider", default="volcengine",
        choices=list(PROVIDERS.keys()),
        help="LLM provider (the API endpoint). All compared models "
             "are served by this single provider.",
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        help=(
            "Whitespace-separated list of model names to compare. "
            "All must be served by --provider. Example: "
            "--models deepseek-v4-flash deepseek-r1 qwen3-max"
        ),
    )
    p.add_argument(
        "--num_games", "--num-games", type=balanced_game_count, default=6,
        help="Number of games per (model_a, model_b) cell, balanced "
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
# Cell runner (independent of Bench.runner's run_cell)
# ---------------------------------------------------------------------------
def run_cell(
    env_id: str,
    memory_item: str,
    model_a: str,
    model_b: str,
    provider: str,
    num_games: int,
    temperature: float,
    out_dir: str,
    *,
    policy_text: str | None = None,
) -> Dict:
    """
    Play `num_games` games between two model names served by the
    same LLM provider. Both sides run in `non_thinking_pgt` mode.

    Result row keys mirror runner.py's summary.csv schema so
    downstream consumers (e.g. plot.py) work unchanged.
    """
    strategy_prompt = policy_text if policy_text is not None else build_policy(env_id, memory_item)
    client_a = make_llm_client(provider, model=model_a, temperature=temperature)
    client_b = make_llm_client(provider, model=model_b, temperature=temperature)

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
        else:
            p0_c, p1_c = client_b, client_a
        before_a, before_b = _snapshot(client_a), _snapshot(client_b)
        result = play_one_matchup(
            env_id=env_id,
            p0_prompt=strategy_prompt,
            p1_prompt=strategy_prompt,
            p0_client=p0_c,
            p1_client=p1_c,
            p0_include_thinking=True,
            p1_include_thinking=True,
        )
        after_a, after_b = _snapshot(client_a), _snapshot(client_b)
        for i in range(4):
            tokens_a[i] += after_a[i] - before_a[i]
            tokens_b[i] += after_b[i] - before_b[i]
        r0, r1 = result["rewards"][0], result["rewards"][1]
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

        # Per-game JSON. We append the provider name to the file name
        # so that a future per-provider run is unambiguous.
        fname = (
            f"{env_id}__{memory_item}__provider={provider}__"
            f"p0={quote(model_a, safe='')}__p1={quote(model_b, safe='')}__"
            f"side={'p0_first' if not swap else 'p1_first'}__game{game_idx:02d}.json"
        )
        out = dict(result)
        out["provider"] = provider
        out["model_p0"] = model_a if not swap else model_b
        out["model_p1"] = model_b if not swap else model_a
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
        "memory_item": memory_item,
        "matchup_type": "model_comparison",
        "thinking_mode": FIXED_THINKING_MODE,
        "provider": provider,
        "model_p0": model_a,
        "model_p1": model_b,
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
        f"prompt_tokens_{model_a}": tokens_a[0],
        f"completion_tokens_{model_a}": tokens_a[1],
        f"reasoning_tokens_{model_a}": tokens_a[2],
        f"call_count_{model_a}": tokens_a[3],
        f"prompt_tokens_{model_b}": tokens_b[0],
        f"completion_tokens_{model_b}": tokens_b[1],
        f"reasoning_tokens_{model_b}": tokens_b[2],
        f"call_count_{model_b}": tokens_b[3],
        "total_prompt_tokens": tokens_a[0] + tokens_b[0],
        "total_completion_tokens": tokens_a[1] + tokens_b[1],
        "total_reasoning_tokens": tokens_a[2] + tokens_b[2],
        "total_calls": tokens_a[3] + tokens_b[3],
        "total_wall_seconds": round(wall_seconds, 2),
    }


def main() -> int:
    args = parse_args()

    # Validate model list.
    models = list(args.models)
    if len(models) < 2:
        raise SystemExit(f"--models requires at least 2 names; got {models}")
    if len(models) != len(set(models)):
        raise SystemExit(f"--models must be unique; got {models}")

    # Validate provider.
    if args.provider not in PROVIDERS:
        raise SystemExit(
            f"Unknown provider {args.provider!r}. Supported: {list(PROVIDERS.keys())}"
        )

    group = load_group_or_exit(args.memory_group, args.game)
    policy_text = group.policies[args.memory_item] if group else None
    if group is None and args.game not in _ITEM_REGISTRY.get(args.memory_item, {}):
        raise SystemExit(
            f"No '{args.memory_item}' item registered for env '{args.game}'."
        )

    # Output dir: {output_dir}/{env}/{memory_item}/model_comparison/<ts>/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(
        args.output_dir, args.game, args.memory_item, "model_comparison", ts
    )
    os.makedirs(run_dir, exist_ok=False)

    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({
            **vars(args), "fixed_thinking_mode": FIXED_THINKING_MODE,
            "memory_group_id": group.group_id if group else None,
            "memory_group_sha256": group.content_sha256 if group else None,
        },
                  f, indent=2)

    cells = list(combinations(models, 2))
    total_cells = len(cells)

    print("=" * 70)
    print("Bench: Model Comparison (fixed thinking mode = non_thinking_pgt)")
    print("=" * 70)
    print(f"Game       : {args.game}")
    print(f"Memory item: {args.memory_item}")
    print(f"Provider   : {args.provider}")
    print(f"Models     : {models}")
    print(f"Cells      : {total_cells}  ({cells})")
    print(f"Games/cell : {args.num_games}  (balanced 50/50 first-mover)")
    print(f"Temperature: {args.temperature}")
    print(f"Output dir : {run_dir}")
    print("=" * 70)

    rows: List[Dict] = []
    bar = _tqdm(total=total_cells, desc="model_comparison",
                disable=not _HAS_TQDM or args.no_tqdm)
    try:
        for model_a, model_b in cells:
            bar.set_description(f"{model_a} vs {model_b}")
            print(f"\n  --- Cell: p0={model_a} vs p1={model_b} ---")
            try:
                row = run_cell(
                    env_id=args.game,
                    memory_item=args.memory_item,
                    model_a=model_a,
                    model_b=model_b,
                    provider=args.provider,
                    num_games=args.num_games,
                    temperature=args.temperature,
                    out_dir=run_dir,
                    policy_text=policy_text,
                )
            except Exception as e:
                print(f"    ! Cell failed: {type(e).__name__}: {e}")
                bar.update(1)
                continue
            rows.append(row)
            print(
                f"    {model_a}:{row['wins_p0']}  {model_b}:{row['wins_p1']}  "
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
            "env_id", "memory_item", "matchup_type", "thinking_mode",
            "provider", "model_p0", "model_p1", "num_games",
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
