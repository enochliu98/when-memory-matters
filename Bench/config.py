"""Shared CLI settings and explicit Memory Item groups."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


PAPER_GAMES = (
    "Nim-v0", "TicTacToe-v0", "ConnectFour-v0",
    "KuhnPoker-v0", "Othello-v0", "Breakthrough-v0",
)
SUPPORTED_GAMES = (*PAPER_GAMES, "UltimateTicTacToe-v0", "LiarsDice-v0")
MEMORY_FAMILIES = ("trajectory", "insight", "reasoning_pattern")
MODE_CONFIG = {
    "thinking": {"thinking_enabled": True, "include_thinking": False},
    "non_thinking_pgt": {"thinking_enabled": False, "include_thinking": True},
    "non_thinking": {"thinking_enabled": False, "include_thinking": False},
}
DEFAULT_OUTPUT_DIR = "runs/bench"


def balanced_game_count(value: str) -> int:
    count = int(value)
    if count <= 0 or count % 2:
        raise argparse.ArgumentTypeError("game count must be a positive even integer")
    return count


@dataclass(frozen=True)
class MemoryGroup:
    group_id: str
    policies: dict[str, str]
    content_sha256: str


def load_memory_group(path: str, game: str) -> MemoryGroup:
    """Load one complete set of authored Language Policy bodies for a Game."""
    data = Path(path).read_bytes()
    group = json.loads(data)
    if not isinstance(group, dict) or group.keys() != {"game", "group_id", "policies"}:
        raise ValueError("a group must have game, group_id, and policies fields")
    if group["game"] != game:
        raise ValueError(f"memory group is for {group['game']!r}, not {game!r}")
    group_id, policies = group["group_id"], group["policies"]
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("group_id must be a nonempty string")
    if not isinstance(policies, dict) or set(policies) != set(MEMORY_FAMILIES):
        raise ValueError("policies must contain trajectory, insight, and reasoning_pattern")
    if any(not isinstance(policy, str) or not policy.strip() for policy in policies.values()):
        raise ValueError("every policy must be a nonempty string")
    return MemoryGroup(group_id, policies, sha256(data).hexdigest())


def load_group_or_exit(path: str | None, game: str) -> MemoryGroup | None:
    if path is None:
        return None
    try:
        return load_memory_group(path, game)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot load memory group: {exc}") from exc
