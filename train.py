import torch
import torch.optim as optim
import torch.nn.functional as F
from model import NoGoNet
from data import GameDataSet
from torch.utils.data import DataLoader


def loss_fn(model: NoGoNet, s, pi, z):
    b = len(pi)
    pi = pi.reshape((b, 81))
    p, v = model.forward(s)
    loss1 = -torch.sum(torch.log(p)*pi)/b
    loss2 = F.mse_loss(v.reshape((b)), z)
    return loss1, loss2


def train(model=None, save_name="model_1", previous="random", epoches=50, lr=2e-3, batch_size=512):
    if model is None:
        model = NoGoNet(scale=4).cuda()
    else:
        assert isinstance(model, str)
        model = torch.load("models/{}.pt".format(model)).cuda()
    dl_train = DataLoader(GameDataSet(model_name=previous),
                          batch_size=batch_size, shuffle=True, num_workers=4)
    dl_eval = DataLoader(GameDataSet(model_name=previous, train=False),
                         batch_size=batch_size, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    best_loss = 1000.
    for e in range(epoches):
        el_p = 0.
        el_v = 0.
        total_n = 0
        model.train()
        for i, (s, p, z) in enumerate(dl_train):
            if i*batch_size > 50000:
                break
            s, p, z = s.cuda(), p.cuda(), z.cuda()
            optimizer.zero_grad()
            loss_p, loss_v = loss_fn(model, s, p, z)
            loss = loss_p+loss_v
            loss.backward()
            el_p += len(z)*loss_p
            el_v += len(z)*loss_v
            total_n += len(z)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            el_p_eval = 0.
            el_v_eval = 0.
            total_n_eval = 0
            for i, (s, p, z) in enumerate(dl_eval):
                if i*batch_size > 10000:
                    break
                s, p, z = s.cuda(), p.cuda(), z.cuda()
                loss_p, loss_v = loss_fn(model, s, p, z)
                el_p_eval += len(z)*loss_p
                el_v_eval += len(z)*loss_v
                total_n_eval += len(z)
        if el_v_eval/total_n_eval < best_loss:
            best_loss = el_v_eval/total_n_eval
            torch.save(model, "models/{}.pt".format(save_name))
        print("epoch:{} loss_p:{:.4f} loss_v:{:.4f} loss_p_eval:{:.4f} loss_v_eval:{:.4f}".format(
            e, el_p/total_n, el_v/total_n, el_p_eval/total_n_eval, el_v_eval/total_n_eval))


if __name__ == "__main__":
    train()
