from math import log
import numpy as np

from nogo import get_legal_actions


class Evaluator:
    """评估器，需要实现一个eval函数"""

    def eval(self, board_A, board_B, legal_actions) -> tuple[np.ndarray, float]:
        """输入棋盘，输出每个位置的p以及对当前局面的评价v"""
        return np.random.random((9, 9))+.01, .0


evaluator = Evaluator()  # 神经网络，或者什么类似的东西，用于指导MCTS
c_puct = 0.5


class _TreeNode:
    """包含了W、N、父子结点以及评估结果"""

    def __init__(self, parent, action, board_A=None, board_B=None) -> None:
        self.parent = parent
        self.w = .0
        self.n = 0
        if parent is None:
            self.board_A = board_A
            self.board_B = board_B
        else:
            self.board_A = np.copy(parent.board_B)
            self.board_B = np.copy(parent.board_A)
            x, y = action
            self.board_B[x][y] = 1
        action_A = get_legal_actions(self.board_A, self.board_B)
        action_B = get_legal_actions(self.board_B, self.board_A)
        if len(action_B) == 0 or len(action_A) == 0:
            if len(action_A) == 0:
                self.v = -1
            else:
                self.v = 1
            self.actions = []
            self.children = []
            self.p = np.zeros((9, 9))
            return
        self.actions = action_A
        self.children = [None]*len(self.actions)
        self.p, self.v = evaluator.eval(
            self.board_A, self.board_B, self.actions)


def _select_and_expand_and_evaluate(t: _TreeNode) -> _TreeNode:
    if len(t.actions) == 0:
        return t
    log_N = log(t.n)
    max_qu = -2.
    max_arg = 0
    for i in range(len(t.children)):
        c = t.children[i]
        x, y = t.actions[i]
        n = c.n if c is not None else 0
        q = -c.w/n if c is not None else 0  # w代表了子节点的v总和，代表对对手的有利程度，这边需要反过来
        qu = q+c_puct*t.p[x][y]*log_N/(1.+n)
        if qu > max_qu:
            max_qu = qu
            max_arg = i
    if t.children[max_arg] is None:
        nt = _TreeNode(t, t.actions[max_arg])
        t.children[max_arg] = nt
        return nt
    return _select_and_expand_and_evaluate(t.children[max_arg])


def _backup(t: _TreeNode) -> None:
    v = t.v
    while t is not None:
        t.w += v
        t.n += 1
        v *= -1  # 交换
        t = t.parent


def mcts(board_A, board_B, max_N) -> np.ndarray:
    """输入棋盘以及最大搜索次数，给出每个位置被访问次数"""
    root = _TreeNode(None, None, board_A, board_B)
    _backup(root)
    for _ in range(max_N):
        t = _select_and_expand_and_evaluate(root)
        _backup(t)
    result = np.zeros((9, 9))
    for i in range(len(root.children)):
        x, y = root.actions[i]
        result[x][y] = root.children[i].n if root.children[i] is not None else 0
    return result


if __name__ == "__main__":
    board_A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    board_B = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
        ]
    )
    print(mcts(board_A, board_B, 1000))
