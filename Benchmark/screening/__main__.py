"""
Demo: Run the GameScreener on zero-sum games.

Usage:
    python -m Benchmark.screening --games TicTacToe-v0 --llm volcengine
    python -m Benchmark.screening --games all --llm volcengine --num-games 6
"""

import argparse
import os

from Bench.config import PAPER_GAMES, balanced_game_count

# Thinking-mode definitions.
MODE_CONFIG = {
    "thinking":         {"thinking_enabled": True,  "include_thinking": False, "output_format_style": "simple"},
    "non_thinking_pgt": {"thinking_enabled": False, "include_thinking": True,  "output_format_style": "thinking_aware"},
    "non_thinking":     {"thinking_enabled": False, "include_thinking": False, "output_format_style": "simple"},
}
COMPARE_MODES = list(MODE_CONFIG.keys())


def _parse_llm_spec(spec: str):
    """Parse an LLM spec string into (provider, model).

    Formats:
        "volcengine"           -> ("volcengine", None)   — use provider default model
        "volcengine:deepseek-r1" -> ("volcengine", "deepseek-r1")  — explicit model
        "deepseek"             -> ("deepseek", None)
        "deepseek:deepseek-chat" -> ("deepseek", "deepseek-chat")
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.strip(), model.strip()
    return spec.strip(), None


def _build_optimal_opponent(env_id: str, llm_client=None):
    """Build the optimal opponent for a given game.

    - 有闭式解的游戏 (Nim / KuhnPoker / LiarsDice) 走规则对手
    - TTT / C4 / Breakthrough 用 minimax; Othello 用 MCTS (评估函数信噪比过低)
    - 没有搜索配置的游戏, 返回 None (触发 LLM-as-Optimal fallback)
    """
    from .opponents.nim import NimOptimal
    from .opponents.kuhnpoker import KuhnPokerOptimal
    from .opponents.liarsdice import LiarsDiceOptimal
    from .opponents.optimal_search import SearchOptimalOpponent, _SEARCH_CONFIG

    _RULE_MAP = {
        "Nim-v0": NimOptimal(),
        "KuhnPoker-v0": KuhnPokerOptimal(),
        "LiarsDice-v0": LiarsDiceOptimal(),
    }
    if env_id in _RULE_MAP:
        return _RULE_MAP[env_id]
    if env_id in _SEARCH_CONFIG:
        return SearchOptimalOpponent(env_id)
    print(f"  [Info] No programmatic optimal opponent for {env_id}. "
          f"Will use LLM-as-Optimal fallback.")
    return None


def _make_client(spec: str, temperature: float, thinking_enabled: bool):
    """Create a single LLM client from a provider[:model] spec."""
    if not spec:
        return None
    try:
        from langpsro.utils import make_llm_client
        provider, model = _parse_llm_spec(spec)
        kwargs = dict(temperature=temperature, thinking_enabled=thinking_enabled)
        if model:
            kwargs["model"] = model
        client = make_llm_client(provider, **kwargs)
        print(f"    client: provider={provider}, model={client.model}, "
              f"thinking_enabled={thinking_enabled}")
        return client
    except ValueError as e:
        raise SystemExit(f"Cannot initialize LLM client: {e}") from e


def _run_screening(
    args,
    mode_name: str,
    games: list,
    random_opponent,
    game_wrapper_cls,
    parent_dir: str = "",
):
    """Run screening for a single thinking mode, persist, and print results.

    Args:
        parent_dir: in --compare-thinking mode, the shared parent directory
                    under which a <mode_name>/ subdir is created.
                    In single-mode, leave empty — the dir is derived from
                    args.save_dir or auto-generated.
    """
    mode_cfg = MODE_CONFIG[mode_name]
    print(f"\n{'='*70}")
    print(f"  Thinking mode: {mode_name}  "
          f"(thinking_enabled={mode_cfg['thinking_enabled']}, "
          f"include_thinking={mode_cfg['include_thinking']})")
    print(f"{'='*70}")

    # Build LLM clients for this mode.
    llm_client = _make_client(args.llm, temperature=0.3,
                               thinking_enabled=mode_cfg["thinking_enabled"])
    optimal_llm_client = _make_client(args.optimal_llm, temperature=0.1,
                                       thinking_enabled=mode_cfg["thinking_enabled"])

    # Build screener for this mode.
    from . import GameScreener
    screener = GameScreener(
        llm_client=llm_client,
        optimal_llm_client=optimal_llm_client,
        num_games_vs_optimal=args.num_games,
        num_evolvability_games=args.num_games,
        record_games=not args.no_game_records,
        thinking_mode=mode_name,
    )

    # Run screening for each game.
    reports = {}
    game_records_map = {}
    for env_id in games:
        gw = game_wrapper_cls(env_id=env_id) if game_wrapper_cls else None
        optimal_opponent = _build_optimal_opponent(env_id, llm_client)

        report = screener.screen(
            env_id,
            random_opponent=random_opponent,
            optimal_opponent=optimal_opponent,
            game_wrapper=gw,
            optimal_llm_client=optimal_llm_client,
        )
        reports[env_id] = report.to_dict()
        if report.game_records:
            game_records_map[env_id] = report.game_records
        print(report.summary())

    # Persistence
    if args.save_dir != "":
        from .reporting import (
            make_run_dir, save_reports_json, save_summary_csv,
            save_plots, save_game_records,
        )
        if parent_dir:
            mode_dir = os.path.join(parent_dir, mode_name)
        else:
            mode_dir = args.save_dir or make_run_dir()

        os.makedirs(mode_dir, exist_ok=True)
        json_path = save_reports_json(reports, mode_dir)
        csv_path = save_summary_csv(reports, mode_dir)
        print(f"\n  [{mode_name}] Artifacts in: {mode_dir}")
        print(f"    json:  {json_path}")
        print(f"    csv:   {csv_path}")
        if not args.no_plots:
            paths = save_plots(reports, mode_dir)
            if paths and "overview" in paths:
                print(f"    overview:{paths['overview']}")
            if paths:
                per_game = [v for k, v in paths.items() if k.startswith("per_game::")]
                if per_game:
                    print(f"    per-game:{os.path.dirname(per_game[0])}/  ({len(per_game)} files)")
        if game_records_map:
            counts = save_game_records(game_records_map, mode_dir)
            total_games = sum(counts.values())
            print(f"    games:  {os.path.join(mode_dir, 'games')}/  "
                  f"({total_games} game records across {len(counts)} envs)")

    # Summary table.
    _print_summary(games, reports)

    return reports, game_records_map


def _print_summary(games, reports):
    """Print the two summary tables (dimension scores + matchup results)."""
    print(f"\n  {'='*58}")
    print("  Summary Table")
    print(f"  {'='*58}")
    print(f"  {'Game':28s} {'Total':>6s}  {'NonSat':>6s} {'Evolv':>6s} {'Cost':>6s}")
    print(f"  {'-'*54}")
    for env_id in games:
        r = reports[env_id]
        dims = r["dimensions"]
        def fmt(d):
            return 'S' if d['skipped'] else str(d['score'])
        print(
            f"  {env_id:28s} "
            f"{r['total_score']:>5.1f}  "
            f"{fmt(dims['non_saturation']):>5s}  "
            f"{fmt(dims['evolvability']):>6s}  "
            f"{fmt(dims['cost']):>5s}"
        )

    print(f"\n  {'='*58}")
    print("  Matchup Results  (胜-负-平, 胜场差)")
    print(f"  {'='*58}")
    for env_id in games:
        dims = reports[env_id]["dimensions"]
        ns_raw = dims.get("non_saturation", {}).get("raw_metrics", {})
        ev_raw = dims.get("evolvability", {}).get("raw_metrics", {})
        ns_skipped = dims.get("non_saturation", {}).get("skipped", False)
        ev_skipped = dims.get("evolvability", {}).get("skipped", False)
        ns_str = "skipped" if ns_skipped else (
            f"{ns_raw.get('win_count', 0)}-{ns_raw.get('loss_count', 0)}-"
            f"{ns_raw.get('draw_count', 0)}, margin {ns_raw.get('margin', 0):+d}"
        )
        ev_str = "skipped" if ev_skipped else (
            f"{ev_raw.get('expert_win_count', 0)}-{ev_raw.get('naive_win_count', 0)}-"
            f"{ev_raw.get('draw_count', 0)}, margin {ev_raw.get('margin', 0):+d}"
        )
        print(f"  {env_id:22s} NonSat(LLM-Opt): {ns_str:22s} Evolv(Exp-Nai): {ev_str}")


def main():
    parser = argparse.ArgumentParser(
        description="GameScreener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LLM spec format:  provider[:model]
  Examples:
    --llm volcengine                        (default model: deepseek-v4-flash)
    --llm volcengine:deepseek-r1            (specific model on volcengine)
    --llm deepseek                          (default model: deepseek-chat)
    --llm deepseek:deepseek-reasoner        (specific model on deepseek)
    --optimal-llm volcengine:deepseek-r1    (stronger model as optimal opponent)

Thinking modes:
    --thinking-mode {thinking,non_thinking_pgt,non_thinking}
    --compare-thinking   run all 3 modes sequentially, saving each under
                         <save-dir>/<mode>/
        """,
    )
    parser.add_argument("--games", nargs="+", default=["TicTacToe-v0"],
                        help="Games to screen (or 'all' for preset list)")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM for the player, format: provider[:model] "
                             "(e.g. volcengine, volcengine:deepseek-v4-flash)")
    parser.add_argument("--optimal-llm", type=str, default=None,
                        help="LLM for optimal opponent, format: provider[:model] "
                             "(e.g. volcengine:deepseek-r1). "
                             "Used for games without programmatic optimal opponents.")
    parser.add_argument("--num-games", type=balanced_game_count, default=6,
                        help="Games per matchup")
    parser.add_argument("--output", type=str, default=None,
                        help="(Legacy) Output JSON path. Prefer --save-dir.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save reports/plots into. Defaults "
                             "to a fresh runs/screening_<timestamp>/ folder. "
                             "Pass an empty string to disable persistence.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip rendering matplotlib plots (JSON/CSV still saved).")
    parser.add_argument("--no-game-records", action="store_true",
                        help="Skip saving per-game trajectories (games/ dir). "
                             "By default each played game is saved as its own JSON.")
    parser.add_argument("--thinking-mode", type=str, default="thinking",
                        choices=list(MODE_CONFIG.keys()),
                        help="Thinking mode for the LLM player. Default: thinking.")
    parser.add_argument("--compare-thinking", action="store_true",
                        help="Run all three thinking modes sequentially, "
                             "saving each under <save-dir>/<mode>/.")
    args = parser.parse_args()

    games = PAPER_GAMES if args.games == ["all"] else args.games

    # Build reusable game infrastructure (opponents, wrapper).
    random_opponent = game_wrapper_cls = None
    if args.llm:
        from .opponents import RandomOpponent
        from .utils import TextArenaGame
        random_opponent = RandomOpponent()
        game_wrapper_cls = TextArenaGame

    if args.compare_thinking:
        from .reporting import make_run_dir
        parent_dir = ""
        if args.save_dir != "":
            parent_dir = args.save_dir or make_run_dir(prefix="screening_compare")

        for mode_name in COMPARE_MODES:
            _run_screening(args, mode_name, games,
                           random_opponent, game_wrapper_cls,
                           parent_dir=parent_dir)
        if parent_dir:
            print(f"\n  All comparison results saved under: {parent_dir}/")
    else:
        _run_screening(args, args.thinking_mode, games,
                       random_opponent, game_wrapper_cls)

    # Legacy --output support (single mode only).
    if args.output and not args.compare_thinking:
        # Reports were the last thing _run_screening returned; but we don't
        # have them here.  Legacy path is preserved only for non-compare runs.
        # The user should migrate to --save-dir.
        print("  [Info] --output is legacy; use --save-dir instead.")


if __name__ == "__main__":
    main()
