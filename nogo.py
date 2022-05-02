from platform import system
from ctypes import CDLL, c_char_p, Structure, c_int32, byref, c_uint8
import numpy as np
import torch

if system() == "Windows":
    nogo_lib = CDLL("./nogo.dll")
elif system() == "Linux":
    nogo_lib = CDLL("./nogo.so")
else:
    raise Exception("不受支持的操作系统")
_c_get_legal_actions = nogo_lib.get_legal_actions


class _LegalActionResult(Structure):
    _fields_ = [
        ("num_s", c_int32),
        ("num_r", c_int32),
        ("res_s", c_uint8*81),
        ("res_r", c_uint8*81),
    ]


def get_legal_actions(board_A: torch.Tensor, board_B: torch.Tensor) -> tuple[list, list]:
    """
    返回A和B玩家可落子点集
    board_A/board_B:9*9棋盘，1代表有子，0代表无子
    """
    b = (board_A-board_B).numpy()
    b = b.astype(np.int8).ctypes.data_as(c_char_p)
    res = _LegalActionResult()
    _c_get_legal_actions(b, byref(res))
    res_a = [(res.res_s[i]//9, res.res_s[i] % 9) for i in range(res.num_s)]
    res_b = [(res.res_r[i]//9, res.res_r[i] % 9) for i in range(res.num_r)]
    return res_a, res_b


class Action(tuple[int, int]):
    pass


class Status():
    def __init__(self, board_A=None, board_B=None, action=None) -> None:
        self.board_A = torch.zeros(
            (9, 9)) if board_A is None else board_A.clone()
        self.board_B = torch.zeros(
            (9, 9)) if board_B is None else board_B.clone()
        if action is not None:
            self.board_B[action[0]][action[1]] = 1
        self.actions_A, self.actions_B = get_legal_actions(
            self.board_A, self.board_B)
        self.actions = self.actions_A
        self.terminate = len(self.actions_A) == 0 or len(self.actions_B) == 0
        self.win = False if not self.terminate else len(self.actions_A) != 0
        # 如果终局的话当前方是否获胜

    def next_status(self, action: Action):
        return Status(self.board_B, self.board_A, action)

    def tensor(self) -> torch.Tensor:
        return torch.stack((self.board_A, self.board_B)).float()


if __name__ == "__main__":
    print(Status())
    board_A = torch.Tensor([
        [1., 1., 0., 0., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 1., 1., 1., 1., 1., 1.],
        [1., 0., 0., 1., 0., 0., 1., 0., 1.],
        [0., 0., 1., 0., 0., 1., 1., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 1., 0., 0.]])
    board_B = torch.Tensor([
        [0., 0., 1., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 1., 0., 1.],
        [1., 0., 0., 1., 1., 0., 1., 1., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 1., 1., 0., 1., 0.],
        [1., 0., 0., 1., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 1., 1., 1., 0.],
        [1., 0., 1., 1., 0., 1., 1., 0., 1.],
        [0., 0., 1., 1., 1., 0., 0., 1., 1.]])
    print(get_legal_actions(board_A, board_B))
