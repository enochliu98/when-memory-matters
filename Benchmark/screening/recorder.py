"""
Per-game trajectory recorder.

The screener's evaluators each play a batch of games (non_saturation vs random
& vs optimal, evolvability baseline & expert). The aggregate win/loss stats
feed the dimension scores, but the *play-by-play* of every game is thrown
away by `play_multiple_games`.

This module captures those per-game trajectories so they can be persisted to
disk for later analysis. Each record is self-contained: the move sequence
plus the final cumulative observation (which, because TextArena observations
are cumulative, contains the full board history of the game).

A `GameRecorder` is created per `screen()` call and threaded through the
evaluators into `TextArenaGame.play_multiple_games`, which calls `record()`
once per finished game. The records are then attached to the `ScreeningReport`
and written to `runs/<run>/games/<env_id>/<matchup>_gameNN.json` by
`reporting.save_game_records`.

Record schema
-------------
    {
      "env_id":     "TicTacToe-v0",
      "matchup":    "llm_vs_optimal",         # llm_vs_optimal | evolv_expert
      "game_idx":   0,                         # index within this matchup's batch
      "p0_label":   "llm",                     # who played as env player 0
      "p1_label":   "random",                  # who played as env player 1
      "winner":     0,                         # env player id (0/1), or -1 for draw
      "rewards":    [1.0, -1.0],
      "num_steps":  7,
      "steps":      [{"step": 1, "player": 0, "action": "[4]"}, ...],
      "final_obs":  "..."                      # full cumulative observation at game end
    }

From `p0_label`/`p1_label` + `winner` you can always recover "did the LLM win
this game": the LLM's label is the matchup's first agent ("llm" / "expert" /
"naive"), and env player 0 always moves first in these games.
"""

from typing import Dict, List, Optional


class GameRecorder:
    """Accumulates per-game trajectory records during a single screen() call."""

    def __init__(self):
        self.records: List[Dict] = []

    def record(
        self,
        env_id: str,
        matchup: Optional[str],
        game_idx: int,
        p0_label: str,
        p1_label: str,
        winner: int,
        rewards: List[float],
        num_steps: int,
        steps: List[Dict],
        final_obs: str,
    ) -> None:
        self.records.append({
            "env_id": env_id,
            "matchup": matchup,
            "game_idx": game_idx,
            "p0_label": p0_label,
            "p1_label": p1_label,
            "winner": winner,
            "rewards": list(rewards),
            "num_steps": num_steps,
            "steps": [
                {"step": i + 1, "player": s["player"], "action": s["action"]}
                for i, s in enumerate(steps)
            ],
            "final_obs": final_obs,
        })

    def __len__(self) -> int:
        return len(self.records)
