from mcts import MonteCarolTree, MultiProcessNNEvaluatorGroup, MultiProcessNNEvaluator
from nogo import Action, Status
from model import NoGoNet
from torch.multiprocessing import Process, Queue
from random import random
import torch
import time


def _play(evaluator: MultiProcessNNEvaluator, plays: int, resQ: Queue, idx: int, tempreature: float):
    torch.manual_seed(idx)
    for g in range(plays):
        s = Status()
        mem_s, mem_p, mem_z = [], [], []
        while not s.terminate:
            deep_search = random() < 0.25
            mct = MonteCarolTree(s, evaluator)
            mct.search(600 if deep_search else 100)
            a = mct.get_action(tempreature)
            mem_s.append(s.tensor().clone() if deep_search else None)
            mem_p.append(mct.get_nmap().clone() if deep_search else None)
            mem_z.append(None)
            s = s.next_status(a)
        z = 1. if s.win else -1.
        print(
            "worker {:03d} completed game {} with result {}".format(idx, g, z))
        l = len(mem_z)
        for i in range(l):
            mem_z[l-i-1] = z
            z = -z
        for i in range(l):
            if mem_s[i] is None:
                continue
            resQ.put((mem_s[i], mem_p[i], mem_z[i]))


def self_play(model: NoGoNet, num_workers=16, batch_size=4, total_play=64, tempreature=1.0):
    mpe = MultiProcessNNEvaluatorGroup(model, num_workers, batch_size)
    resQ = Queue()
    workers = [Process(target=_play,
                       args=[mpe.evaluators[i], total_play//num_workers, resQ, i, tempreature]) for i in range(num_workers)]
    for w in workers:
        w.start()
    working = True
    mem_s, mem_p, mem_z = [], [], []
    while working:
        time.sleep(0.1)
        working = False
        for w in workers:
            working = working or w.is_alive()
        while not resQ.empty():
            s, p, z = resQ.get()
            mem_s.append(s)
            mem_p.append(p)
            mem_z.append(z)
    print("done!")
    pass  # todo:保存数据


if __name__ == "__main__":
    self_play(NoGoNet())
