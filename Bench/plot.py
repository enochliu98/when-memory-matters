"""
Plotting helpers for the LangPSRO Bench.

Reads `summary.csv` from a run directory and produces a single PNG
with three subplots:

  1. **Win-rate heatmap** (3x3): for each (mode_p0, mode_p1) cell,
     shows the win rate of mode_p0. Diagonal cells are filled as
     draws (0.5 by convention — they were never played, so we
     visually mark them as "draw" rather than 0/1).
  2. **Per-mode token totals** (bar): for each mode, sum
     `prompt_tokens_<mode>` + `completion_tokens_<mode>` across all
     cells. Bars coloured by mode.
  3. **Win-rate vs token cost scatter**: each (mode_a, mode_b) cell
     plotted as a point at (win_rate_p0, total_prompt_tokens). Shows
     the tradeoff between strength and cost.

Usage (CLI):
    python Bench/plot.py path/to/run_dir
    python Bench/plot.py path/to/run_dir --out compare.png
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# matplotlib is heavy and optional — warn instead of failing hard if
# not installed, but require it for the actual plot.
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no display required)
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


THINKING_MODES = ["thinking", "non_thinking_pgt", "non_thinking"]


def load_summary(csv_path: str) -> List[Dict]:
    """Read summary.csv and return a list of row dicts.

    Both row schemas (3x3 thinking matrix and vs-baseline) are coerced.
    Integer fields (counts, tokens) are detected by exact match or
    by a suffix like `_tokens_<mode>`, `call_count_<mode>`, or
    `_<role>`. Float fields (rates, durations) are matched
    explicitly. Unknown columns are left as strings.

    The detection is suffix-based (not substring-based) so that a
    column like `win_rate_policy` (float) is NOT mistakenly matched
    by an `n_` substring rule.
    """
    # Integer-valued column suffixes produced by runner.py.
    int_suffixes = (
        "_tokens_thinking", "_tokens_non_thinking",
        "_tokens_non_thinking_pgt", "_tokens_policy",
        "_tokens_baseline",
        "call_count_thinking", "call_count_non_thinking",
        "call_count_non_thinking_pgt", "call_count_policy",
        "call_count_baseline",
    )
    int_exact = {"wins_p0", "wins_p1", "wins_policy", "wins_baseline",
                 "n_p0_first", "n_p1_first",
                 "n_policy_first", "n_baseline_first",
                 "draws", "num_games",
                 "total_prompt_tokens", "total_completion_tokens",
                 "total_reasoning_tokens", "total_calls"}
    float_exact = {"win_rate_p0", "win_rate_p1", "draw_rate",
                   "avg_reward_p0", "avg_game_length",
                   "total_wall_seconds",
                   "win_rate_policy", "win_rate_baseline"}

    def _is_int_col(k: str) -> bool:
        return k in int_exact or any(k.endswith(s) for s in int_suffixes)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k in list(r.keys()):
            v = r[k]
            if v == "":
                continue
            if _is_int_col(k):
                try:
                    r[k] = int(v)
                except (ValueError, TypeError):
                    pass
            elif k in float_exact:
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
    return rows


def build_matrix(rows: List[Dict], modes: List[str]
                 ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Build the 3x3 matrices needed for plotting.

    Returns:
        win_matrix:    3x3, win_rate_p0 (NaN if not played)
        draw_matrix:   3x3, draw_rate (NaN if not played)
        token_matrix:  3x3, total_prompt_tokens for that cell
        call_matrix:   3x3, total_calls for that cell
    """
    n = len(modes)
    win = np.full((n, n), np.nan)
    draw = np.full((n, n), np.nan)
    tok = np.full((n, n), np.nan)
    calls = np.full((n, n), np.nan)
    for r in rows:
        # The 3x3 matrix only contains `matchup_type == "matrix"` rows.
        # `vs_baseline` rows have a different schema (role-anchored)
        # and are rendered in a separate subplot, so we skip them
        # here. We also use `.get()` so a row missing mode_p0/mode_p1
        # (e.g. an old-format row) is silently skipped instead of
        # raising.
        if r.get("matchup_type", "matrix") != "matrix":
            continue
        m0, m1 = r.get("mode_p0"), r.get("mode_p1")
        if m0 not in modes or m1 not in modes or m0 == m1:
            continue
        i = modes.index(m0)
        j = modes.index(m1)
        win[i, j] = r["win_rate_p0"]
        draw[i, j] = r["draw_rate"]
        tok[i, j] = r["total_prompt_tokens"]
        calls[i, j] = r["total_calls"]
    # Display the opposite orientation without adding games to the data.
    for i in range(n):
        for j in range(n):
            if i != j and np.isnan(win[i, j]) and not np.isnan(win[j, i]):
                win[i, j] = 1.0 - win[j, i] - draw[j, i]
                draw[i, j] = draw[j, i]
                tok[i, j] = tok[j, i]
                calls[i, j] = calls[j, i]
    return win, draw, tok, calls


def per_mode_token_totals(rows: List[Dict], modes: List[str]) -> Dict[str, Dict[str, int]]:
    """Sum token counts over observed, unique interface pairings."""
    out: Dict[str, Dict[str, int]] = {
        m: {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0} for m in modes
    }
    for r, left, right in played_matrix_rows(rows, modes):
        for m in (left, right):
            out[m]["prompt"] += int(r.get(f"prompt_tokens_{m}", 0) or 0)
            out[m]["completion"] += int(r.get(f"completion_tokens_{m}", 0) or 0)
            out[m]["reasoning"] += int(r.get(f"reasoning_tokens_{m}", 0) or 0)
            out[m]["calls"] += int(r.get(f"call_count_{m}", 0) or 0)
    return out


def played_matrix_rows(rows, modes):
    seen = set()
    for row in rows:
        left, right = row.get("mode_p0"), row.get("mode_p1")
        if row.get("matchup_type", "matrix") != "matrix":
            continue
        if left not in modes or right not in modes or left == right:
            continue
        key = row.get("env_id"), row.get("memory_item"), frozenset((left, right))
        if key in seen:
            continue
        seen.add(key)
        yield row, left, right


def capability_cost_points(rows, modes, totals):
    """Return (draw-adjusted outcome score, completion tokens per call)."""
    points = {mode: [0.0, 0] for mode in modes}
    for row, left, right in played_matrix_rows(rows, modes):
        n = row["num_games"]
        for mode, wins in ((left, row["wins_p0"]), (right, row["wins_p1"])):
            points[mode][0] += wins + 0.5 * row["draws"]
            points[mode][1] += n
    return {
        mode: (score / n, totals[mode]["completion"] / totals[mode]["calls"])
        for mode, (score, n) in points.items()
        if n > 0 and totals[mode]["calls"] > 0
    }


def plot_summary(run_dir: str, out_path: Optional[str] = None) -> str:
    """
    Render the 3-subplot summary figure for a run directory.

    Returns the path of the PNG written.
    """
    if not _HAS_MPL:
        raise SystemExit(
            "matplotlib is required for plotting. Install with:\n"
            "  pip install matplotlib numpy"
        )

    csv_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"No summary.csv found in {run_dir}")
    rows = load_summary(csv_path)
    if not rows:
        raise SystemExit(f"summary.csv is empty: {csv_path}")

    if not _HAS_NP:
        raise SystemExit(
            "numpy is required for plotting. Install with:\n"
            "  pip install numpy"
        )

    # All 3 subplots use only the 3 thinking modes. We explicitly
    # exclude baseline (vs_baseline) rows from plotting — those
    # carry a different schema and are stored in summary.csv for
    # downstream analysis but not rendered here.
    matrix_modes = THINKING_MODES
    plot_modes = THINKING_MODES
    win, draw, tok, calls = build_matrix(rows, matrix_modes)
    totals = per_mode_token_totals(rows, plot_modes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Bench: {os.path.basename(run_dir)}  "
                 f"({len(rows)} cells, "
                 f"{sum(int(r.get('num_games', 0) or 0) for r in rows)} games total)",
                 fontsize=12)

    # --- Subplot 1: Win-rate heatmap ---
    ax = axes[0]
    im = ax.imshow(win, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(matrix_modes)))
    ax.set_xticklabels(matrix_modes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrix_modes)))
    ax.set_yticklabels(matrix_modes, fontsize=9)
    ax.set_xlabel("p1 mode")
    ax.set_ylabel("p0 mode")
    ax.set_title("Win rate of row vs column\n"
                 "Dash: unplayed or unavailable")
    # Annotate each cell.
    for i in range(len(matrix_modes)):
        for j in range(len(matrix_modes)):
            if np.isnan(win[i, j]):
                txt = "--"
                color = "black"
            else:
                txt = f"{win[i, j]:.0%}"
                color = ("white" if win[i, j] < 0.4 or win[i, j] > 0.6
                         else "black")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="win rate")

    # --- Subplot 2: Per-mode token totals ---
    ax = axes[1]
    x = np.arange(len(plot_modes))
    width = 0.2
    p = [totals[m]["prompt"] for m in plot_modes]
    c = [totals[m]["completion"] for m in plot_modes]
    r_ = [totals[m]["reasoning"] for m in plot_modes]
    calls_ = [totals[m]["calls"] for m in plot_modes]
    ax.bar(x - 1.5 * width, p, width, label="prompt", color="#4C72B0")
    ax.bar(x - 0.5 * width, c, width, label="completion", color="#DD8452")
    ax.bar(x + 0.5 * width, r_, width, label="reasoning (subset of completion)", color="#55A868")
    ax2 = ax.twinx()
    ax2.bar(x + 1.5 * width, calls_, width, label="calls (right axis)",
            color="#C44E52", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_modes, fontsize=9)
    ax.set_xlabel("mode")
    ax.set_ylabel("tokens (prompt / completion / reasoning)")
    ax2.set_ylabel("LLM call count", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax.set_title("Per-mode token totals + call count")
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    # --- Subplot 3: Per-interface capability vs token cost ---
    # Each point represents one thinking mode, NOT one (mode_a, mode_b)
    # cell. This lets the user compare the three reasoning methods
    # directly in win-rate-vs-cost space.
    #
    # X: avg win rate of this method when it sits at p0 (average over
    #    all upper-triangle cells where mode_p0 == this mode).
    # Y: total prompt tokens this method consumed across all cells.
    ax = axes[2]
    colors = {"thinking": "#4C72B0", "non_thinking_pgt": "#DD8452",
              "non_thinking": "#55A868"}

    # Per-method scatter aggregation. Each matrix cell accumulates
    # token totals for BOTH mode_p0 and mode_p1 into the respective
    # Compare the same outcome and cost units as the paper.
    for mode, (score, cost) in capability_cost_points(rows, plot_modes, totals).items():
        ax.scatter(
            score, cost,
            s=180, alpha=0.85,
            color=colors.get(mode, "gray"),
            edgecolors="black", linewidth=1.0,
        )
        # Label the point with the mode name + a small breakdown
        ax.annotate(
            f"{mode}\n({totals[mode]['calls']} calls)",
            (score, cost),
            textcoords="offset points", xytext=(10, 8),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Outcome score (win + half draw)")
    ax.set_ylabel("Completion tokens per call")
    ax.set_title("Per-method tradeoff: strength vs cost")
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[m], label=m) for m in plot_modes]
    ax.legend(handles=handles, fontsize=8, loc="best")

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(run_dir, "summary.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot a Bench run summary (win-rate, tokens, tradeoff)."
    )
    p.add_argument(
        "run_dir",
        help="Run directory containing summary.csv (e.g. Bench/results/bench_insight_20260101_120000)",
    )
    p.add_argument(
        "--out", default=None,
        help="Output PNG path (default: <run_dir>/summary.png)",
    )
    args = p.parse_args()

    out = plot_summary(args.run_dir, args.out)
    print(f"Plot written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
