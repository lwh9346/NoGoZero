from typing import Tuple
import torch
import os
import os.path as path
from torch.utils.data import Dataset
from random import random

_s = None
_z = None
_p = None


class GameDataSet(Dataset):
    def __init__(self, model_name="random", train=True) -> None:
        super().__init__()
        global _s, _z, _p
        print("loading data")
        if _s is None:
            if path.exists("self_play/{}_cache.pth".format(model_name)):
                print("loading data from cache")
                d = torch.load("self_play/{}_cache.pth".format(model_name))
                _s = d["s"]
                _p = d["p"]
                _z = d["z"]
        if _s is None:
            _s, _p, _z = [], [], []
            game_data_files = os.listdir("self_play/{}/".format(model_name))
            for f in game_data_files:
                data = torch.load("self_play/{}/{}".format(model_name, f))
                _s.append(data["s"])
                _p.append(data["p"])
                _z.append(data["z"])
            _s = torch.cat(_s).float()
            _p = torch.cat(_p).float()
            _z = torch.cat(_z).float()
            for i in range(_p.shape[0]):
                sum = torch.sum(_p[i])
                assert sum != 0
                _p[i] /= sum
            print("saving data to cache")
            torch.save({"s": _s, "p": _p, "z": _z},
                       "self_play/{}_cache.pth".format(model_name))
        if train:
            t = int(len(_s)*0.8)
            self.s = _s[:t]
            self.z = _z[:t]
            self.p = _p[:t]
        else:
            t = int(len(_s)*0.8)
            self.s = _s[t:]
            self.z = _z[t:]
            self.p = _p[t:]
        self.train = train
        print("data loaded")

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, p, z = self.s[i], self.p[i], self.z[i]
        if not self.train:
            return s, p, z
        if random() > 0.5:
            s[0] = torch.flipud(s[0])
            s[1] = torch.flipud(s[1])
            s[2] = torch.flipud(s[2])
            s[3] = torch.flipud(s[3])
            p = torch.flipud(p)
        if random() > 0.5:
            s[0] = torch.fliplr(s[0])
            s[1] = torch.fliplr(s[1])
            s[2] = torch.fliplr(s[2])
            s[3] = torch.fliplr(s[3])
            p = torch.fliplr(p)
        return s, p, z
