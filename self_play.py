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
            mct.search(160 if deep_search else 32)
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


def self_play(model: NoGoNet, save_to: str, num_workers=8, batch_size=8, total_play=64, tempreature=1.0):
    print("开始自我对局")
    begin = time.time()
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
    print("正在保存数据")
    mem_s = torch.stack(mem_s).type(dtype=torch.int8)
    mem_p = torch.stack(mem_p).type(dtype=torch.uint8)  # for n < 255
    mem_z = torch.tensor(mem_z).type(dtype=torch.int8)
    data = {"s": mem_s, "p": mem_p, "z": mem_z}
    torch.save(data, save_to)
    print("数据已保存至：{}".format(save_to))
    print("自我对局总计用时{:.1f}秒".format(time.time()-begin))


if __name__ == "__main__":
    self_play(NoGoNet())
