from typing import Tuple
import torch
import os
from torch.utils.data import Dataset, DataLoader


class GameDataSet(Dataset):
    def __init__(self, model_name="random", train=True) -> None:
        super().__init__()
        s = []
        p = []
        z = []
        print("loading data")
        game_data_files = os.listdir("self_play/{}/".format(model_name))
        for f in game_data_files:
            data = torch.load("self_play/{}/{}".format(model_name, f))
            s.append(data["s"])
            p.append(data["p"])
            z.append(data["z"])
        s = torch.cat(s).float()
        p = torch.cat(p).float()
        z = torch.cat(z).float()
        for i in range(p.shape[0]):
            sum = torch.sum(p[i])
            assert sum != 0
            p[i] /= sum
        if train:
            t = int(len(s)*0.8)
            s = s[:t]
            z = z[:t]
            p = p[:t]
        else:
            t = int(len(s)*0.8)
            s = s[t:]
            z = z[t:]
            p = p[t:]
        self.s = s
        self.z = z
        self.p = p
        print("data loaded")

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.s[i], self.p[i], self.z[i]
