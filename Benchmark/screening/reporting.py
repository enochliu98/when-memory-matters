"""
Persistence + plotting for GameScreener results.

For each `python -m screening` run, we materialise everything under a
single timestamped directory:

    runs/screening_YYYYMMDD_HHMMSS/
    ├── reports.json          # full to_dict() per game
    ├── summary.csv           # one row per game, easy to diff/share
    ├── per_game/
    │   └── <env_id>.png      # 4-panel per-game breakdown
    ├── games/
    │   └── <env_id>/
    │       └── <matchup>_gameNN.json   # one file per played game
    └── overview.png          # cross-game comparison (totals + dimensions)

Design notes:
  - We deliberately keep matplotlib as a soft dependency. If it isn't
    installed, JSON/CSV still get saved; only the plots are skipped.
  - Plot styling is intentionally plain — no seaborn, no custom colour
    maps — so the output reads cleanly in dark- and light-mode terminals.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

# Matches the dimension order used in ScreeningReport.summary().
DIMENSION_ORDER = ["non_saturation", "evolvability", "cost"]

# Stable per-dimension colour palette (Matplotlib's default-cycle subset).
DIMENSION_COLORS = {
    "non_saturation": "#1f77b4",  # blue
    "evolvability":   "#2ca02c",  # green
    "cost":           "#ff7f0e",  # orange
    "total":          "#7f7f7f",  # grey
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def make_run_dir(base_dir: str = "runs", prefix: str = "screening") -> str:
    """Create runs/<prefix>_<timestamp>/ and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Persistence (JSON + CSV)
# ---------------------------------------------------------------------------

def save_reports_json(reports: Dict[str, dict], run_dir: str) -> str:
    """Dump every report's to_dict() into reports.json."""
    path = os.path.join(run_dir, "reports.json")
    with open(path, "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    return path


def save_summary_csv(reports: Dict[str, dict], run_dir: str) -> str:
    """One row per game with totals + per-dimension scores + key cost stats.

    Skipped dimensions get an empty cell so spreadsheets still align.
    """
    path = os.path.join(run_dir, "summary.csv")
    fieldnames = [
        "env_id", "total_score", "verdict",
        "non_saturation_score", "evolvability_score", "cost_score",
        # Non-saturation: LLM vs Optimal — 胜/负/平 + 胜场差
        "ns_llm_wins", "ns_optimal_wins", "ns_draws", "ns_margin",
        # Evolvability: Expert vs Naive — 胜/负/平 + 胜场差
        "ev_expert_wins", "ev_naive_wins", "ev_draws", "ev_margin",
        # Cost
        "avg_game_length", "tokens_per_game",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for env_id, rep in reports.items():
            dims = rep.get("dimensions", {})
            ns = dims.get("non_saturation", {})
            ev = dims.get("evolvability", {})
            co = dims.get("cost", {})
            ns_raw = ns.get("raw_metrics", {})
            ev_raw = ev.get("raw_metrics", {})
            co_raw = co.get("raw_metrics", {})
            row = {
                "env_id": env_id,
                "total_score": rep.get("total_score"),
                "verdict": _verdict_for(rep.get("total_score", 0.0)),
                "non_saturation_score": "" if ns.get("skipped") else ns.get("score"),
                "evolvability_score":   "" if ev.get("skipped") else ev.get("score"),
                "cost_score":           "" if co.get("skipped") else co.get("score"),
                "ns_llm_wins":      ns_raw.get("win_count"),
                "ns_optimal_wins":  ns_raw.get("loss_count"),
                "ns_draws":         ns_raw.get("draw_count"),
                "ns_margin":        ns_raw.get("margin"),
                "ev_expert_wins":   ev_raw.get("expert_win_count"),
                "ev_naive_wins":    ev_raw.get("naive_win_count"),
                "ev_draws":         ev_raw.get("draw_count"),
                "ev_margin":        ev_raw.get("margin"),
                "avg_game_length":  co_raw.get("avg_game_length") or ns_raw.get("avg_game_length"),
                "tokens_per_game":  ns_raw.get("tokens_per_game"),
            }
            writer.writerow(row)
    return path


def _verdict_for(total: float) -> str:
    if total >= 4.0: return "STRONGLY_RECOMMENDED"
    if total >= 3.0: return "RECOMMENDED"
    if total >= 2.0: return "NOT_RECOMMENDED"
    return "AVOID"


def _safe_name(s: str) -> str:
    """Filesystem-safe segment (for env_id / matchup in filenames)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def save_game_records(
    game_records: Dict[str, List[dict]],
    run_dir: str,
) -> Dict[str, int]:
    """Write each recorded game to its own JSON file.

    Layout:  <run_dir>/games/<env_id>/<matchup>_gameNN.json

    Args:
        game_records: {env_id: [record, ...]} — the per-game trajectories
            attached to each ScreeningReport.
        run_dir: the run directory (a `games/` subdir is created inside).

    Returns: {env_id: number_of_files_written} for the summary printout.
    """
    counts: Dict[str, int] = {}
    games_root = os.path.join(run_dir, "games")
    os.makedirs(games_root, exist_ok=True)

    for env_id, records in game_records.items():
        if not records:
            continue
        env_dir = os.path.join(games_root, _safe_name(env_id))
        os.makedirs(env_dir, exist_ok=True)
        written = 0
        for rec in records:
            matchup = _safe_name(rec.get("matchup") or "game")
            idx = rec.get("game_idx", 0)
            fname = f"{matchup}_game{idx:02d}.json"
            path = os.path.join(env_dir, fname)
            with open(path, "w") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            written += 1
        counts[env_id] = written

    return counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(reports: Dict[str, dict], run_dir: str) -> Optional[Dict[str, str]]:
    """Render an overview plot + one per-game plot.

    Returns a dict {kind: path}, or None if matplotlib isn't installed.
    All plotting failures are caught and logged — JSON/CSV are the
    canonical record, plots are nice-to-have.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("  [Info] matplotlib not installed; skipping plots.")
        return None

    paths: Dict[str, str] = {}
    try:
        paths["overview"] = _plot_overview(reports, run_dir)
    except Exception as e:
        print(f"  [Warn] overview plot failed: {e}")

    per_game_dir = os.path.join(run_dir, "per_game")
    os.makedirs(per_game_dir, exist_ok=True)
    for env_id, rep in reports.items():
        try:
            paths[f"per_game::{env_id}"] = _plot_per_game(env_id, rep, per_game_dir)
        except Exception as e:
            print(f"  [Warn] per-game plot for {env_id} failed: {e}")
    return paths


def _plot_overview(reports: Dict[str, dict], run_dir: str) -> str:
    """Two-panel cross-game comparison.

    Top:    grouped bar chart — per-dimension SCORES (1-5) for each game.
    Bottom: bar chart of TOTAL weighted scores with verdict-tier shading.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    games = list(reports.keys())
    n = len(games)
    if n == 0:
        raise ValueError("No reports to plot")

    fig, (ax_dims, ax_total) = plt.subplots(
        2, 1, figsize=(max(8, 1.6 * n + 2), 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --- Top panel: per-dimension grouped bars ---
    width = 0.25
    x = np.arange(n)
    for i, dim_name in enumerate(DIMENSION_ORDER):
        scores = []
        for env_id in games:
            d = reports[env_id]["dimensions"].get(dim_name, {})
            scores.append(0 if d.get("skipped") else d.get("score", 0))
        offset = (i - 1) * width
        bars = ax_dims.bar(
            x + offset, scores, width,
            label=dim_name, color=DIMENSION_COLORS[dim_name],
        )
        for bar, score, env_id in zip(bars, scores, games):
            d = reports[env_id]["dimensions"].get(dim_name, {})
            if d.get("skipped"):
                ax_dims.text(bar.get_x() + bar.get_width() / 2, 0.15,
                              "S", ha="center", va="bottom",
                              fontsize=8, color="grey")
    ax_dims.set_xticks(x)
    ax_dims.set_xticklabels(games, rotation=20, ha="right")
    ax_dims.set_ylim(0, 5.4)
    ax_dims.set_yticks(range(0, 6))
    ax_dims.set_ylabel("Dimension score (1–5)")
    ax_dims.set_title("Per-dimension scores per game")
    ax_dims.grid(axis="y", linestyle=":", alpha=0.4)
    ax_dims.legend(loc="upper right", fontsize=9, frameon=False)

    # --- Bottom panel: total weighted score with verdict bands ---
    totals = [reports[g]["total_score"] for g in games]
    bars = ax_total.bar(x, totals, color=DIMENSION_COLORS["total"])
    for bar, total in zip(bars, totals):
        ax_total.text(bar.get_x() + bar.get_width() / 2, total + 0.08,
                       f"{total:.1f}", ha="center", va="bottom", fontsize=9)
    # Verdict-band shading.
    ax_total.axhspan(4.0, 5.05, color="green",  alpha=0.07, zorder=0)
    ax_total.axhspan(3.0, 4.0,  color="yellow", alpha=0.07, zorder=0)
    ax_total.axhspan(2.0, 3.0,  color="orange", alpha=0.07, zorder=0)
    ax_total.axhspan(0,   2.0,  color="red",    alpha=0.07, zorder=0)
    # Verdict threshold lines, with labels INSIDE the chart on the left.
    for y, lab in [(4.0, "Strongly recommended (≥4.0)"),
                    (3.0, "Recommended (≥3.0)"),
                    (2.0, "Not recommended (≥2.0)")]:
        ax_total.axhline(y, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)
        # Place label on the left, in-axis (no clipping).
        ax_total.text(0.02, y + 0.04, lab, fontsize=8,
                       ha="left", va="bottom", color="grey",
                       transform=ax_total.get_yaxis_transform())
    ax_total.set_xticks(x)
    ax_total.set_xticklabels(games, rotation=20, ha="right")
    ax_total.set_ylim(0, 5.05)
    ax_total.set_ylabel("Total weighted score")
    ax_total.set_title("Total weighted scores (verdict bands shaded)")
    ax_total.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(run_dir, "overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_per_game(env_id: str, report: dict, out_dir: str) -> str:
    """Four-panel breakdown for a single game.

    [0,0] Dimension scores (horizontal bars)
    [0,1] Win-rate triplet (vs random, vs optimal, draw rate)
    [1,0] Evolvability: baseline vs expert win-rate, plus skill gap
    [1,1] Cost: avg game length + tokens-per-game (dual axis)
    """
    import matplotlib.pyplot as plt

    dims = report.get("dimensions", {})
    ns = dims.get("non_saturation", {})
    ev = dims.get("evolvability", {})
    co = dims.get("cost", {})
    ns_raw = ns.get("raw_metrics", {})
    ev_raw = ev.get("raw_metrics", {})
    co_raw = co.get("raw_metrics", {})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"{env_id} — total {report.get('total_score', 0):.1f}/5.0  "
        f"({_verdict_for(report.get('total_score', 0))})",
        fontsize=12, y=0.995,
    )

    # --- 1. Per-dimension scores ---
    ax = axes[0, 0]
    names = []; scores = []; colors = []
    for d_name in DIMENSION_ORDER:
        d = dims.get(d_name, {})
        names.append(d_name)
        scores.append(0 if d.get("skipped") else d.get("score", 0))
        colors.append(DIMENSION_COLORS[d_name] if not d.get("skipped") else "lightgrey")
    bars = ax.barh(names, scores, color=colors)
    for bar, d_name in zip(bars, names):
        d = dims.get(d_name, {})
        label = "SKIPPED" if d.get("skipped") else (
            f"{d.get('score')}/5  (w={d.get('weight', 0):.0%})"
        )
        # Place label just past the bar end (or inside if the bar is short
        # because the dimension was skipped).
        x_text = max(bar.get_width(), 0.3) + 0.1
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha="left", fontsize=9, color="black")
    # Reserve a margin on the right for the in-line label.
    ax.set_xlim(0, 6.5)
    ax.set_xlabel("Score (1–5)")
    ax.set_title("Per-dimension scores")
    ax.invert_yaxis()  # non_saturation on top
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # --- 2. Non-saturation: LLM vs Optimal — 胜/负/平 + 胜场差 ---
    ax = axes[0, 1]
    if ns.get("skipped"):
        ax.text(0.5, 0.5, "non_saturation skipped",
                ha="center", va="center", color="grey", fontsize=10,
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        llm_wins = ns_raw.get("win_count", 0)
        opt_wins = ns_raw.get("loss_count", 0)
        draws = ns_raw.get("draw_count", 0)
        n = max(1, ns_raw.get("num_games", 1))
        margin = ns_raw.get("margin", 0)
        labels = ["LLM 胜", "Optimal 胜\n(LLM 负)", "平"]
        values = [llm_wins, opt_wins, draws]
        bars = ax.bar(labels, values, color=["#3182bd", "#de2d26", "#bdbdbd"])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(values) * 1.25 + 0.5)
        ax.set_ylabel("局数")
        src = ns_raw.get("optimal_source")
        title = f"LLM vs Optimal — 胜场差 {margin:+d} (margin_rate {margin / n:+.0%})"
        if src:
            title += f"  [optimal: {src}]"
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # --- 3. Evolvability: Expert vs Naive — 胜/负/平 + 胜场差 ---
    ax = axes[1, 0]
    if ev.get("skipped"):
        ax.text(0.5, 0.5, "evolvability skipped",
                ha="center", va="center", color="grey", fontsize=10,
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        exp_wins = ev_raw.get("expert_win_count", 0)
        naiv_wins = ev_raw.get("naive_win_count", 0)
        draws = ev_raw.get("draw_count", 0)
        n = max(1, ev_raw.get("num_games", 1))
        margin = ev_raw.get("margin", 0)
        labels = ["Expert 胜", "Naive 胜\n(Expert 负)", "平"]
        values = [exp_wins, naiv_wins, draws]
        bars = ax.bar(labels, values, color=["#2ca02c", "#de2d26", "#bdbdbd"])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(values) * 1.25 + 0.5)
        ax.set_ylabel("局数")
        ax.set_title(
            f"Expert vs Naive — 胜场差 {margin:+d} (margin_rate {margin / n:+.0%})"
        )
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # --- 4. Cost: dual-axis ---
    ax = axes[1, 1]
    if co.get("skipped"):
        ax.text(0.5, 0.5, "cost skipped",
                ha="center", va="center", color="grey", fontsize=10,
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        avg_len = co_raw.get("avg_game_length", 0)
        tpg = co_raw.get("tokens_per_game", 0)
        # Use two distinct x positions so the bars don't sit on top of
        # each other. Each metric owns its own twin-axis bar.
        x_pos = [0, 1]
        labels = ["avg steps / game", "tokens / game"]
        bar1 = ax.bar([x_pos[0]], [avg_len], width=0.6, color="#ff7f0e")
        ax.text(x_pos[0], avg_len, f" {avg_len:.1f}", ha="center",
                 va="bottom", fontsize=9, color="#ff7f0e")
        ax.set_ylabel("Steps / game", color="#ff7f0e")
        ax.tick_params(axis="y", labelcolor="#ff7f0e")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel("")
        if tpg and tpg > 0:
            ax2 = ax.twinx()
            bar2 = ax2.bar([x_pos[1]], [tpg], width=0.6, color="#9467bd")
            ax2.text(x_pos[1], tpg, f" {tpg:,.0f}", ha="center",
                      va="bottom", fontsize=9, color="#9467bd")
            ax2.set_ylabel("Tokens / game", color="#9467bd")
            ax2.tick_params(axis="y", labelcolor="#9467bd")
        title = "Cost from real rollouts"
        src = co_raw.get("score_source")
        if src:
            title += f"  ({src})"
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe = env_id.replace("/", "_").replace(":", "_")
    path = os.path.join(out_dir, f"{safe}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
