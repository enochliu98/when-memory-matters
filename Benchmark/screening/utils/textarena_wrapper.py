"""
Thin wrapper around TextArena for the GameScreener.

Provides a unified interface for playing games with LLM agents
and collecting game statistics, independent of the LangPSRO codebase.
"""

import textarena as ta
from typing import Optional, List, Dict, Tuple, Callable


class TextArenaGame:
    """Wrapper for a single TextArena game environment."""

    def __init__(self, env_id: str, num_players: int = 2):
        self.env_id = env_id
        self.num_players = num_players
        self.env: Optional[ta.Env] = None

    def reset(self) -> str:
        """Reset the environment and return the initial observation."""
        self.env = ta.make(self.env_id)
        self.env.reset(num_players=self.num_players)
        _, obs = self.env.get_observation()
        return obs

    def step(self, action: str) -> Tuple[bool, List[float], str]:
        """
        Take a step in the environment.

        Returns:
            (done, rewards, next_observation)

        On a terminal step we grab the final observation BEFORE closing the
        env, so callers can persist the end-of-game board state. (TextArena
        observations are cumulative, so this final obs contains the entire
        move-by-move board history.) If for some reason the obs can't be
        read post-done, we fall back to an empty string.
        """
        if self.env is None:
            raise RuntimeError("Environment not reset")
        done, step_info = self.env.step(action=action)
        if done:
            final_obs = ""
            try:
                _, final_obs = self.env.get_observation()
            except Exception:
                final_obs = ""
            rewards, _ = self.env.close()
            if isinstance(rewards, dict):
                rewards = [rewards.get(i, 0.0) for i in range(self.num_players)]
            elif not isinstance(rewards, list):
                rewards = [rewards]
            return True, rewards, final_obs
        _, obs = self.env.get_observation()
        return False, [0.0] * self.num_players, obs

    def play_game(
        self,
        agent_p0: Callable[[str], str],
        agent_p1: Callable[[str], str],
        max_steps: int = 200,
    ) -> Dict:
        """
        Play a full game between two agents.

        Real env pid 决定轮到谁, 不依赖 step_count 的固定交替.
        非法走子被 env 拒绝后 pid 不变, 不会错误翻转 player_id.

        Returns:
            Dict with keys: winner, rewards, trajectory, num_steps, final_obs
            - `winner`: env player id (0/1) or -1 for draw
            - `trajectory`: list of {"player", "action"} per STEP (走子方)
            - `final_obs`: full cumulative observation at game end
        """
        self.reset()
        trajectory = []
        step_count = 0
        final_obs = ""

        for _ in range(max_steps * 2):  # 给拒绝留余量
            pid, obs = self.env.get_observation()
            agent = agent_p0 if pid == 0 else agent_p1
            action = agent(obs)
            done, rewards, next_obs = self.step(action)
            if done:
                final_obs = next_obs if next_obs else obs
                winner = (
                    0 if rewards[0] > rewards[1]
                    else 1 if rewards[1] > rewards[0]
                    else -1
                )
                trajectory.append({"player": pid, "action": action})
                return {
                    "winner": winner,
                    "rewards": rewards,
                    "trajectory": trajectory,
                    "num_steps": step_count + 1,
                    "final_obs": final_obs,
                }
            # 检查是否被拒绝: pid 不变 → 不记录, 不计数
            new_pid, _ = self.env.get_observation()
            if new_pid == pid:
                continue  # env 拒绝了, 重试
            trajectory.append({"player": pid, "action": action})
            step_count += 1
            obs = next_obs

        return {
            "winner": -1,
            "rewards": [0.0, 0.0],
            "trajectory": trajectory,
            "num_steps": step_count,
            "final_obs": obs,
        }

    def play_multiple_games(
        self,
        agent_p0: Callable[[str], str],
        agent_p1: Callable[[str], str],
        num_games: int,
        swap_first: bool = True,
        recorder=None,
        matchup_label: Optional[str] = None,
        agent_labels: Optional[Tuple[str, str]] = None,
    ) -> Dict:
        """
        Play multiple games, optionally swapping first-move advantage.

        Args:
            recorder: optional GameRecorder; if given, each finished game is
                recorded (env_id / matchup / per-game trajectory).
            matchup_label: tag stored on each recorded game (e.g.
                "llm_vs_random"). Only used when `recorder` is provided.
            agent_labels: (label_for_agent_p0, label_for_agent_p1). Used to
                record who played as env player 0/1 in each game (accounting
                for swap_first). Defaults to ("p0", "p1").

        Returns aggregated statistics (by original agent order, i.e. p0 =
        the first positional agent argument).
        """
        lab0 = agent_labels[0] if agent_labels else "p0"
        lab1 = agent_labels[1] if agent_labels else "p1"

        p0_wins = p1_wins = draws = 0
        total_steps = 0

        for i in range(num_games):
            swapped = swap_first and i >= num_games // 2
            if swapped:
                # env P0 = agent_p1, env P1 = agent_p0
                result = self.play_game(agent_p1, agent_p0)
                env_p0_label, env_p1_label = lab1, lab0
            else:
                result = self.play_game(agent_p0, agent_p1)
                env_p0_label, env_p1_label = lab0, lab1

            w = result["winner"]
            if swapped:
                # env P0 is agent_p1, so an env-P0 win counts for p1.
                if w == 0:
                    p1_wins += 1
                elif w == 1:
                    p0_wins += 1
                else:
                    draws += 1
            else:
                if w == 0:
                    p0_wins += 1
                elif w == 1:
                    p1_wins += 1
                else:
                    draws += 1

            total_steps += result["num_steps"]

            if recorder is not None:
                recorder.record(
                    env_id=self.env_id,
                    matchup=matchup_label,
                    game_idx=i,
                    p0_label=env_p0_label,
                    p1_label=env_p1_label,
                    winner=w,
                    rewards=result["rewards"],
                    num_steps=result["num_steps"],
                    steps=result["trajectory"],
                    final_obs=result["final_obs"],
                )

        return {
            "p0_wins": p0_wins,
            "p1_wins": p1_wins,
            "draws": draws,
            "total_games": num_games,
            "p0_win_rate": p0_wins / num_games,
            "p1_win_rate": p1_wins / num_games,
            "draw_rate": draws / num_games,
            "avg_game_length": total_steps / num_games,
        }
