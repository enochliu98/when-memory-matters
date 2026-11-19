"""
通用 UCT MCTS (Upper Confidence bounds applied to Trees) 搜索引擎。

接口：

    mcts_move(state, iterations, **kwargs) → str

调用方需要传入实现以下协议的 GameState 对象：

    class MyState:
        def get_legal_actions(self) -> List[str]:
        def apply_action(self, action: str) -> "MyState":
        def is_terminal(self) -> Tuple[bool, Optional[int]]:
        def evaluate(self, player: int) -> float:    # rollout 评分用
        def player_to_move(self) -> int:              # 0/1

可选的定制接口 (直接在 state 上定义同名方法):

    def rollout_policy(state) -> str:
        返回 rollout 阶段的默认动作。默认 = uniform random。

    def evaluate_state(state) -> float:
        返回终局评分 (替代 rollout)。默认 = rollout 跑到终局。

用法示例
--------
    from screening.search.mcts import mcts_move
    from screening.search.games.connectfour import ConnectFourState

    state = ConnectFourState.parse_from_obs(obs, my_id)
    action = mcts_move(state, iterations=3000, time_limit=2.0)
"""

import math
import random
import time
from typing import List, Optional

# UCT 探索常数
_DEFAULT_C = 1.414


class MCTSNode:
    """MCTS 树节点。"""

    __slots__ = ("state", "parent", "children", "untried", "visits", "total_score",
                 "player_to_move")

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.untried: List[str] = list(state.get_legal_actions())
        self.visits = 0
        self.total_score = 0.0
        self.player_to_move = state.player_to_move

    @property
    def is_expanded(self) -> bool:
        return len(self.untried) == 0

    @property
    def is_terminal_node(self) -> bool:
        return self.state.is_terminal()[0]


def _uct_score(node: MCTSNode, c: float) -> float:
    """UCT 公式: Q + C * sqrt(log(N_parent) / n)."""
    if node.visits == 0:
        return float("inf")
    if node.parent is None:
        return node.total_score / node.visits
    exploitation = node.total_score / node.visits
    exploration = c * math.sqrt(math.log(node.parent.visits) / node.visits)
    # total_score 是从 parent 视角存入的, 即 child 的值 = -parent 视角
    return -exploitation + exploration


def _default_rollout(state, get_action_fn) -> float:
    """从 state rollout 到终局, 返回当前玩家视角的分值 (win=+1, lose=-1, draw=0)."""
    s = state
    while True:
        terminal, winner = s.is_terminal()
        if terminal:
            if winner is None:
                return 0.0
            return 1.0 if winner == state.player_to_move else -1.0
        actions = s.get_legal_actions()
        if not actions:
            return 0.0
        action = get_action_fn(s)
        s = s.apply_action(action)


def mcts_move(
    state,
    iterations: int = 2000,
    time_limit: Optional[float] = None,
    c: float = _DEFAULT_C,
    verbose: bool = False,
) -> str:
    """通过 UCT MCTS 返回最佳动作。

    Args:
        state:      GameState 对象
        iterations:  最大模拟次数
        time_limit:  时间预算 (秒), 超时即止
        c:           UCT 探索常数
        verbose:     打印搜索统计

    Returns:
        最佳动作字符串
    """
    legal = state.get_legal_actions()
    if not legal:
        return ""

    # 快速赢法检测
    for action in legal:
        child = state.apply_action(action)
        terminal, winner = child.is_terminal()
        if terminal and winner == state.player_to_move:
            return action

    # 获取 rollout 策略
    rollout_policy = getattr(state, "rollout_policy", None)

    def _rollout_action(s) -> str:
        if rollout_policy is not None:
            return rollout_policy(s)
        legal_acts = s.get_legal_actions()
        return random.choice(legal_acts) if legal_acts else ""

    root = MCTSNode(state)
    start = time.time()

    for i in range(iterations):
        if time_limit is not None and i % 50 == 0:
            elapsed = time.time() - start
            if elapsed >= time_limit:
                break

        # SELECT
        node = root
        while node.is_expanded and not node.is_terminal_node and node.children:
            node = max(node.children, key=lambda n: _uct_score(n, c))

        # EXPAND
        if not node.is_terminal_node and not node.is_expanded:
            action = random.choice(node.untried)
            node.untried.remove(action)
            child_state = node.state.apply_action(action)
            child = MCTSNode(child_state, parent=node)
            node.children.append(child)
            node = child

        # ROLLOUT
        terminal, winner = node.state.is_terminal()
        if terminal:
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if winner == node.state.player_to_move else -1.0
        else:
            # Use evaluate_state if available (deterministic eval), else rollout
            eval_state = getattr(node.state, "evaluate_state", None)
            if eval_state is not None:
                value = eval_state(node.state)
            else:
                value = _default_rollout(node.state, _rollout_action)

        # BACKPROPAGATE
        while node is not None:
            node.visits += 1
            node.total_score += value
            value = -value  # 翻转视角
            node = node.parent

    # 选择访问最多的子节点
    best_action = legal[0]
    best_visits = -1
    for child in root.children:
        if child.visits > best_visits:
            best_visits = child.visits
            # 找出对应的 action
            for a in legal:
                if state.apply_action(a) == child.state:
                    best_action = a
                    break

    if verbose:
        print(f"[MCTS] iter={min(iterations, len(range(iterations)))}"
              f" ({time.time() - start:.2f}s), "
              f"children={len(root.children)}, "
              f"best={best_action}({best_visits} visits)")
        for child in root.children:
            a = "?"
            for a_cand in legal:
                if state.apply_action(a_cand) == child.state:
                    a = a_cand
                    break
            print(f"  {a}: {child.visits} visits, "
                  f"Q={child.total_score / max(child.visits, 1):.3f}")

    return best_action
