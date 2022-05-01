import numpy as np

_cxy = [(-1, 0), (0, -1), (1, 0), (0, 1)]
_points = [(i//9, i % 9) for i in range(81)]


def get_legal_actions(board_A, board_B) -> list:
    """
    返回A玩家可落子点集
    board_A/board_B:9*9棋盘，1代表有子，0代表无子
    """
    sole_airs_B = set()
    sole_airs_A = set()
    not_sole_airs_A = set()
    visited = set()
    for x, y in _points:
        if (x, y) in visited:
            continue
        if board_A[x][y]:
            members, airs = _bfs_group(board_A, board_B, x, y)
            for m in members:
                visited.add(m)
            for a in airs:
                if len(airs) == 1:
                    sole_airs_A.add(a)
                else:
                    not_sole_airs_A.add(a)
        if board_B[x][y]:
            members, airs = _bfs_group(board_B, board_A, x, y)
            for m in members:
                visited.add(m)
            if len(airs) == 1:
                for a in airs:
                    sole_airs_B.add(a)
    result = []
    for p in _points:
        if p in visited:
            continue
        if p in sole_airs_B:
            continue
        if p in not_sole_airs_A:
            result.append(p)
            continue
        if p in sole_airs_A:
            x, y = p
            for dx, dy in _cxy:
                fx, fy = x+dx, y+dy
                if not _in_board(fx, fy):
                    continue
                if board_A[fx][fy] == 0 and board_B[fx][fy] == 0:
                    result.append((x, y))
                    break
            continue
        x, y = p
        not_dead_point = False
        for dx, dy in _cxy:
            fx, fy = x+dx, y+dy
            if _in_board(fx, fy) and board_A[fx][fy] == 0 and board_B[fx][fy] == 0:
                not_dead_point = True
        if not_dead_point:
            result.append(p)
    return result


def get_legal_actions_AB(board_A, board_B) -> tuple(list, list):
    """
    返回A和B玩家可落子点集
    board_A/board_B:9*9棋盘，1代表有子，0代表无子
    """
    sole_airs_A = set()
    sole_airs_B = set()
    not_sole_airs_A = set()
    not_sole_airs_B = set()
    visited = set()
    for x, y in _points:
        if (x, y) in visited:
            continue
        if board_A[x][y]:
            members, airs = _bfs_group(board_A, board_B, x, y)
            for m in members:
                visited.add(m)
            for a in airs:
                if len(airs) == 1:
                    sole_airs_A.add(a)
                else:
                    not_sole_airs_A.add(a)
        if board_B[x][y]:
            members, airs = _bfs_group(board_B, board_A, x, y)
            for m in members:
                visited.add(m)
            for a in airs:
                if len(airs) == 1:
                    sole_airs_B.add(a)
                else:
                    not_sole_airs_B.add(a)
    result_A = []
    for p in _points:
        if p in visited:
            continue
        if p in sole_airs_B:
            continue
        if p in not_sole_airs_A:
            result_A.append(p)
            continue
        if p in sole_airs_A:
            x, y = p
            for dx, dy in _cxy:
                fx, fy = x+dx, y+dy
                if not _in_board(fx, fy):
                    continue
                if board_A[fx][fy] == 0 and board_B[fx][fy] == 0:
                    result_A.append((x, y))
                    break
            continue
        x, y = p
        not_dead_point = False
        for dx, dy in _cxy:
            fx, fy = x+dx, y+dy
            if _in_board(fx, fy) and board_A[fx][fy] == 0 and board_B[fx][fy] == 0:
                not_dead_point = True
        if not_dead_point:
            result_A.append(p)
    result_B = []
    for p in _points:
        if p in visited:
            continue
        if p in sole_airs_A:
            continue
        if p in not_sole_airs_B:
            result_B.append(p)
            continue
        if p in sole_airs_B:
            x, y = p
            for dx, dy in _cxy:
                fx, fy = x+dx, y+dy
                if not _in_board(fx, fy):
                    continue
                if board_B[fx][fy] == 0 and board_A[fx][fy] == 0:
                    result_B.append((x, y))
                    break
            continue
        x, y = p
        not_dead_point = False
        for dx, dy in _cxy:
            fx, fy = x+dx, y+dy
            if _in_board(fx, fy) and board_B[fx][fy] == 0 and board_A[fx][fy] == 0:
                not_dead_point = True
        if not_dead_point:
            result_B.append(p)
    return result_A, result_B


def _bfs_group(board_A, board_B, x0, y0):
    airs = set()
    visited = set()
    to_visit = [(x0, y0)]
    i = 0
    while i < len(to_visit):
        x, y = to_visit[i]
        i += 1
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in _cxy:
            fx, fy = x+dx, y+dy
            if not _in_board(fx, fy):
                continue
            if board_A[fx][fy] and not (fx, fy) in visited:
                to_visit.append((fx, fy))
    for x, y in visited:
        for dx, dy in _cxy:
            fx, fy = x+dx, y+dy
            if not _in_board(fx, fy):
                continue
            if board_A[fx][fy] == 0 and board_B[fx][fy] == 0:
                airs.add((fx, fy))
    return visited, airs


def _in_board(x, y):
    return x >= 0 and x < 9 and y >= 0 and y < 9


class Status():
    def __init__(self, board_A=None, board_B=None, action=None) -> None:
        self.board_A = np.zeros((9, 9)) if board_A is None else board_A
        self.board_B = np.zeros((9, 9)) if board_B is None else board_B
        if not action is None:
            self.board_B[action[0]][action[1]] = 1
        self.actions_A, self.actions_B = get_legal_actions_AB(board_A, board_B)
        self.actions = self.actions_A
        self.terminate = len(self.actions_A) == 0 or len(self.actions_B) == 0
        self.win = False if not self.terminate else len(self.actions_A) != 0
        # 如果终局的话当前方是否获胜

    def get_next_status(self, action):
        return Status(self.board_B, self.board_A, action)


if __name__ == "__main__":
    board_A = [
        [1., 1., 0., 0., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 1., 1., 1., 1., 1., 1.],
        [1., 0., 0., 1., 0., 0., 1., 0., 1.],
        [0., 0., 1., 0., 0., 1., 1., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 1., 0., 0.]]
    board_B = [
        [0., 0., 1., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 1., 0., 1.],
        [1., 0., 0., 1., 1., 0., 1., 1., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 1., 1., 0., 1., 0.],
        [1., 0., 0., 1., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 1., 1., 1., 0.],
        [1., 0., 1., 1., 0., 1., 1., 0., 1.],
        [0., 0., 1., 1., 1., 0., 0., 1., 1.]]
    print(get_legal_actions(board_A, board_B))
