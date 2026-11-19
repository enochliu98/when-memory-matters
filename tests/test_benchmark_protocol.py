import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from Bench import llm_compare, memory_compare, play, runner
from Bench.config import MEMORY_FAMILIES, PAPER_GAMES, balanced_game_count, load_memory_group
from Benchmark.screening.__main__ import _build_optimal_opponent
from langpsro.envs.textarena_env import TextArenaEnv, extract_actions_from_observation


DEMO = Path(__file__).resolve().parents[1] / "examples/tictactoe_demo_group.json"


def test_group_schema_and_digest(tmp_path):
    group = load_memory_group(str(DEMO), "TicTacToe-v0")
    assert set(group.policies) == set(MEMORY_FAMILIES)
    assert group.group_id == "demo-not-experimental-data"
    assert group.content_sha256 == sha256(DEMO.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not 'Nim-v0'"):
        load_memory_group(str(DEMO), "Nim-v0")
    invalid = json.loads(DEMO.read_text())
    invalid["policies"].pop("trajectory")
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="policies must contain"):
        load_memory_group(str(path), "TicTacToe-v0")


@pytest.mark.parametrize("count", ["0", "-2", "3"])
def test_requires_balanced_games(count):
    from argparse import ArgumentTypeError

    assert balanced_game_count("2") == 2
    with pytest.raises(ArgumentTypeError, match="positive even"):
        balanced_game_count(count)


@pytest.mark.parametrize("game", PAPER_GAMES)
def test_reference_opponent_plays_a_legal_opening(game):
    env = TextArenaEnv(game)
    _, observation = env.reset()
    try:
        legal = extract_actions_from_observation(observation, game)
        assert legal
        move = _build_optimal_opponent(game).select_action(observation, legal)
        assert move in legal
    finally:
        env.env.close()


def test_offline_tictactoe_game():
    x = iter(["Action: [0]", "Action: [1]", "Action: [2]"])
    o = iter(["Action: [4]", "Action: [8]"])
    result = play.play_one_matchup(
        "TicTacToe-v0", "Play as first policy.", "Play as second policy.",
        lambda _: next(x), lambda _: next(o),
    )
    assert result["winner"] == "p0"
    assert [call["parsed_action"] for call in result["llm_calls"]] == [
        "[0]", "[4]", "[1]", "[8]", "[2]",
    ]


def test_truncated_game_is_not_a_draw(monkeypatch):
    env = SimpleNamespace(play_game=lambda **_: SimpleNamespace(done=False))
    monkeypatch.setattr(play, "TextArenaEnv", lambda **_: env)
    with pytest.raises(RuntimeError, match="before a terminal outcome"):
        play.play_one_matchup("TicTacToe-v0", "", "", lambda _: "", lambda _: "")


class StubClient:
    def get_usage_summary(self):
        return {"total": dict.fromkeys(
            ["prompt_tokens", "completion_tokens", "reasoning_tokens", "call_count"],
            0,
        )}


@pytest.mark.parametrize(
    ("mode", "requested", "expected_toggles", "expected_sections"),
    [
        ("thinking", "matched", [True, True], [False, False]),
        ("non_thinking_pgt", "matched", [False, False], [True, True]),
        ("thinking", "non_thinking_pgt", [True, False], [False, True]),
    ],
)
def test_baseline_controls_and_side_swaps(
    tmp_path, monkeypatch, mode, requested, expected_toggles, expected_sections,
):
    configs, matches = [], []

    def client(provider, **kwargs):
        configs.append((provider, kwargs))
        return StubClient()

    def matchup(**kwargs):
        matches.append(kwargs)
        return {"rewards": [1, -1], "trajectory": []}

    monkeypatch.setattr(runner, "make_llm_client", client)
    monkeypatch.setattr(runner, "play_one_matchup", matchup)
    result = runner.run_baseline_matchup(
        "TicTacToe-v0", "reasoning_pattern", mode, 2, 0.3, str(tmp_path),
        model="offline-test-model", policy_text="Reviewed custom policy.", baseline_mode=requested,
    )
    assert [x[1]["thinking_enabled"] for x in configs] == expected_toggles
    assert {x[1]["model"] for x in configs} == {"offline-test-model"}
    assert [matches[0][f"p{i}_include_thinking"] for i in (0, 1)] == expected_sections
    assert [matches[1][f"p{i}_include_thinking"] for i in (1, 0)] == expected_sections
    assert matches[0]["p0_prompt"] == matches[1]["p1_prompt"] == "Reviewed custom policy."
    assert matches[0]["p1_prompt"] == matches[1]["p0_prompt"]
    assert result["baseline_thinking_mode"] == (mode if requested == "matched" else requested)
    assert (result["wins_policy"], result["wins_baseline"], result["draws"]) == (1, 1, 0)


@pytest.mark.parametrize("fail", [False, True])
def test_memory_cli_plays_each_pair_and_logs_group(tmp_path, monkeypatch, fail):
    seen = []
    monkeypatch.setattr(sys, "argv", [
        "wmm-memory-items", "--items", *MEMORY_FAMILIES,
        "--model", "offline-test-model", "--num-games", "2", "--no_tqdm",
        "--output-dir", str(tmp_path), "--memory-group", str(DEMO),
    ])
    monkeypatch.setattr(memory_compare, "resolve_model", lambda provider, model: model)

    def cell(**kwargs):
        seen.append((kwargs["item_a"], kwargs["item_b"]))
        assert kwargs["policies"]["trajectory"].startswith("You are a TicTacToe")
        assert kwargs["model"] == "offline-test-model"
        if fail and len(seen) == 1:
            raise RuntimeError("simulated cell failure")
        return dict(wins_p0=1, wins_p1=1, draws=0, total_prompt_tokens=0, total_wall_seconds=0)

    monkeypatch.setattr(memory_compare, "run_cell", cell)
    assert memory_compare.main() == int(fail)
    assert len(seen) == len(set(seen)) == 3
    info = json.loads(next(tmp_path.rglob("run_config.json")).read_text())
    assert info["memory_group_id"] == "demo-not-experimental-data"
    assert info["memory_group_sha256"] == sha256(DEMO.read_bytes()).hexdigest()
    with next(tmp_path.rglob("summary.csv")).open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 3 - int(fail)


def test_model_id_with_slash_is_safe_in_trace_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_compare, "make_llm_client", lambda *a, **k: StubClient())
    monkeypatch.setattr(llm_compare, "play_one_matchup", lambda **k: {
        "rewards": [1, -1], "trajectory": [],
    })
    llm_compare.run_cell(
        "TicTacToe-v0", "insight", "org/model-a", "org/model-b",
        "volcengine", 2, 0.3, str(tmp_path), policy_text="Same policy.",
    )
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_plot_scores_both_sides_without_counting_mirrors_or_reasoning_twice():
    from Bench.plot import capability_cost_points, per_mode_token_totals

    left, right = "thinking", "non_thinking"
    actual = {
        "mode_p0": left, "mode_p1": right, "num_games": 4,
        "wins_p0": 2, "wins_p1": 1, "draws": 1,
        f"completion_tokens_{left}": 100, f"reasoning_tokens_{left}": 70,
        f"completion_tokens_{right}": 12, f"call_count_{left}": 4,
        f"call_count_{right}": 3,
    }
    mirror = dict(actual, mode_p0=right, mode_p1=left, wins_p0=1, wins_p1=2)
    diagonal = dict(actual, mode_p1=left, wins_p0=0, wins_p1=0, draws=4)
    rows = [actual, mirror, diagonal]
    modes = [left, right]
    totals = per_mode_token_totals(rows, modes)
    assert capability_cost_points(rows, modes, totals) == {
        left: (0.625, 25.0), right: (0.375, 4.0),
    }
