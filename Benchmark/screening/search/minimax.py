"""
通用 Negamax + α-β 剪枝 + 转置表搜索引擎。

与游戏无关的核心接口：

    minimax_move(state, depth, **kwargs) → str

调用方需要传入实现以下协议的 GameState 对象：

    class MyState:
        def get_legal_actions(self) -> List[str]:
        def apply_action(self, action: str) -> "MyState":
        def is_terminal(self) -> Tuple[bool, Optional[int]]:
        def evaluate(self, player: int) -> float:     # depth=0 时调
        def player_to_move(self) -> int:               # 0/1
        def move_key(self, action: str) -> Any: ...     # 可选, move ordering
        def __hash__(self) -> int:                      # 转置表用
        def __eq__(self, other) -> bool:

此外可定义可选方法 evaluate_terminal(state, winner, depth_remaining) → float,
返回 depth-aware 终局分 (默认 +/- 100000 + depth_bonus)。

用法示例
--------
    from screening.search.minimax import minimax_move
    from screening.search.games.tictactoe import TicTacToeState

    state = TicTacToeState.parse_from_obs(obs, my_id)
    action = minimax_move(state, depth=9, use_tt=True)
"""


# 终局分值基数
TERMINAL_WIN = 100_000.0


def minimax_move(
    state,
    depth: int = 6,
    use_tt: bool = True,
    tt=None,
    verbose: bool = False,
) -> str:
    """返回当前 state 下的最佳动作。

    Args:
        state:  GameState 对象
        depth:  搜索深度
        use_tt: 是否启用转置表
        tt:     可选外部 dict(建议跨 move 复用)；内部使用 if use_tt
        verbose: 打印搜索统计

    Returns:
        最佳动作字符串
    """
    legal = state.get_legal_actions()
    if not legal:
        return ""

    trans_table = tt if (use_tt and tt is not None) else ({} if use_tt else None)
    eval_terminal = getattr(state, "evaluate_terminal", None)

    # 1) 快速赢法检测
    for action in legal:
        child = state.apply_action(action)
        terminal, winner = child.is_terminal()
        if terminal and winner == state.player_to_move:
            return action

    # 2) 完整搜索
    best_score = -float("inf")
    best_action = legal[0]
    sort_key = getattr(state, "move_key", None)
    ordered = sorted(legal, key=sort_key, reverse=True) if sort_key else list(legal)
    alpha, beta = -float("inf"), float("inf")

    if verbose:
        nodes_before = len(trans_table) if trans_table else 0

    for action in ordered:
        child = state.apply_action(action)
        opp = 1 - state.player_to_move
        score = -_negamax(child, depth - 1, -beta, -alpha, opp,
                          trans_table, eval_terminal)
        if score > best_score:
            best_score = score
            best_action = action
        if score > alpha:
            alpha = score

    if verbose and trans_table is not None:
        nodes_searched = len(trans_table) - (nodes_before if use_tt else 0)
        print(f"[Minimax] depth={depth}, nodes={nodes_searched}, "
              f"actions={len(legal)}, best={best_action}({best_score:.1f})")

    return best_action


def _negamax(state, depth, alpha, beta, player, tt, eval_terminal):
    """递归 Negamax，返回 player 视角的分值。"""
    # 终局
    terminal, winner = state.is_terminal()
    if terminal:
        if eval_terminal is not None:
            return eval_terminal(state, winner, depth)
        if winner is None:
            return 0.0
        bonus = 10.0 * (depth + 1)
        return TERMINAL_WIN + bonus if winner == player else -(TERMINAL_WIN + bonus)

    if depth <= 0:
        return state.evaluate(player)

    # 转置表查询
    if tt is not None and state in tt:
        tt_score, tt_depth = tt.get(state, (None, -1))
        if tt_score is not None and tt_depth >= depth:
            return tt_score

    legal = state.get_legal_actions()
    if not legal:
        return state.evaluate(player)

    sort_key = getattr(state, "move_key", None)
    ordered = sorted(legal, key=sort_key, reverse=True) if sort_key else list(legal)
    best = -float("inf")
    opp = 1 - player

    for action in ordered:
        child = state.apply_action(action)
        score = -_negamax(child, depth - 1, -beta, -alpha, opp,
                          tt, eval_terminal)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    if tt is not None:
        tt[state] = (best, depth)

    return best
