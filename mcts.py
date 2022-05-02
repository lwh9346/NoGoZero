import torch
import numpy as np

from math import log, sqrt
import multiprocessing as mp
from queue import Queue

from nogo import Status, Action
from model import NoGoNet


class Evaluator:
    """
    评估器，需要实现两个eval函数
    输入的评估状态和输出遵循FIFO的原则
    """

    def start_eval(self, s: Status) -> None:
        """输入状态s，输出每个位置的p以及对当前局面的评价v"""
        pass

    def get_eval_result(self) -> tuple[torch.Tensor, float]:
        """输入状态s，输出每个位置的p以及对当前局面的评价v"""
        p, v = torch.rand((9, 9)), 0.0
        p /= torch.sum(p)
        return p, v


class BasicNNEvaluator(Evaluator):
    def __init__(self, model: NoGoNet, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self._eval_queue = []
        self._results = Queue()

    def start_eval(self, s: Status) -> None:
        self._eval_queue.append(s.tensor())

    @torch.no_grad()
    @torch.cuda.amp.autocast_mode.autocast()
    def get_eval_result(self) -> tuple[torch.Tensor, float]:
        if not self._results.empty():
            return self._results.get()
        self.model.to(self.device)
        ps, vs = self.model(torch.stack(self._eval_queue).to(self.device))
        ps, vs = ps.cpu(), vs.cpu()
        self._eval_queue = []
        for i in range(len(ps)):
            self._results.put((ps[i], float(vs[i])))
        return self._results.get()


class MultiProcessNNEvaluator(Evaluator):
    def __init__(self, evalQ: mp.Queue, resQ: mp.Queue, idx: int) -> None:
        self._idx = idx
        self._eval_queue = evalQ
        self._results = resQ

    def start_eval(self, s: Status) -> None:
        self._eval_queue.put((s.tensor(), self._idx))

    def get_eval_result(self) -> tuple[torch.Tensor, float]:
        return self._results.get()


class MultiProcessNNEvaluatorGroup():
    @staticmethod
    @torch.no_grad()
    @torch.cuda.amp.autocast_mode.autocast()
    def _work_no_stop(evalQ: mp.Queue, resQ: list[mp.Queue], batch_size: int, model: NoGoNet):
        while True:
            e, idx = [], []
            for _ in range(batch_size):
                s, i = evalQ.get()
                e.append(s)
                idx.append(i)
            e = torch.stack(e).cuda()
            ps, vs = model(e)
            ps, vs = ps.cpu(), vs.cpu()
            for i in range(batch_size):
                resQ[idx[i]].put(ps[i], float(vs[i]))

    def __init__(self, model: NoGoNet, num_evaluator: int, batch_size=16) -> None:
        self._evalQ = mp.Queue()
        self._resQ = [mp.Queue() for _ in range(num_evaluator)]
        self.evaluators = [MultiProcessNNEvaluator(
            self._evalQ, self._resQ[i]) for i in range(num_evaluator)]
        self._worker = mp.Process(target=MultiProcessNNEvaluatorGroup._work_no_stop,
                                  args=[self._evalQ, self._resQ, batch_size, model])
        self._worker.start()


class _TreeNode:
    """包含了W、N、父子结点"""

    def __init__(self, parent, s: Status, a: Action) -> None:
        self._s = s
        self.a = a
        self.parent = parent
        self.is_leaf = True
        self.w = .0  # 当前玩家视角
        self.n = 0

    @property
    def s(self) -> Status:
        if self._s is None:
            self._s = self.parent.s.next_status(self.a)
        return self._s


class MonteCarolTree:
    def __init__(self, s: Status, evaluator: Evaluator, c_puct=1.1) -> None:
        assert not s.terminate
        self._root = _TreeNode(None, s, None)
        self._evaluator = evaluator
        self._c_puct = c_puct

    def search(self, max_steps: int) -> None:
        for _ in range(max_steps):
            # select
            t = self._root
            while not t.is_leaf:
                # PUCT
                i = np.argmax([-t.w/t.n+self._c_puct*t.p[i]*sqrt(t.n) /
                               (1+t.children[i].n) for i in range(len(t.children))])
                t = t.children[i]
            # expand and evaluate
            if not t.s.terminate:
                t.is_leaf = False
                t.children = [_TreeNode(t, None, a) for a in t.s.actions]
                self._evaluator.start_eval(t.s.tensor())
                p, v = self._evaluator.get_eval_result()
                v = float(v)
                t.p = [float(p[x][y]) for x, y in t.s.actions]
            else:
                v = 1. if t.s.win else -1.
            # backup
            while t is not None:
                t.w += v
                t.n += 1
                v = -v
                t = t.parent
        self._nmap = torch.zeros((9, 9))
        for i, (x, y) in enumerate(self._root.s.actions):
            self._nmap[x][y] = self._root.children[i].n

    def get_action(self, tempreature=1.0) -> Action:
        w = torch.reshape(self._nmap, (81,))
        u = torch.rand((81,))
        k = u**(tempreature/w)
        a = int(np.argmax(k))
        return a//9, a % 9

    def get_nmap(self) -> torch.Tensor:
        return torch.reshape(self._nmap, (9, 9))


if __name__ == "__main__":
    board_A = torch.tensor(
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
    board_B = torch.tensor(
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
    s = Status(board_A, board_B)
    e = Evaluator()
    mct = MonteCarolTree(s, e)
    import cProfile
    print(cProfile.run("mct.search(800)"))
    print(mct.get_nmap())
    print(mct.get_action())
