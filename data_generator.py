from copy import deepcopy
import os
import sys
import numpy as np
import random
import subprocess
import platform

import torch

import mcts
from model import NoGoNet
import nogo


def _random_mcts_cpp(board_A, board_B, max_N):
    # max_N参数被忽略，固定输出10000步的结果
    input_data = ["C"]*81
    for i in range(81):
        x, y = i//9, i % 9
        if board_A[x][y] == 1:
            input_data[i] = "A"
        if board_B[x][y] == 1:
            input_data[i] = "B"
    input_data.append("\n")
    input_data = "".join(input_data).encode("ASCII")
    cmd = "data_generator.exe" if platform.system() == "Windows" else "./data_generator"
    res = subprocess.run(cmd, input=input_data, capture_output=True)
    res = res.stdout.decode("ASCII")
    res = [int(t) for t in res.split()[:81]]
    res = np.reshape(np.array(res), (9, 9))
    return res


if __name__ == "__main__":
    # 参数列表
    if len(sys.argv) == 5:
        model = sys.argv[1]
        games_to_play = int(sys.argv[2])
        tempreature = float(sys.argv[3])
        rand_seed = int(sys.argv[4])
    else:
        model = "model_1"
        games_to_play = 10
        tempreature = 2.0
        rand_seed = 0
    # 初始化
    """
    try:
        os.makedirs("log/{}".format(model))
    except:
        pass
    log_file = open(
        "log/{}/{}games_{}.log".format(model, games_to_play, rand_seed), "w")
    sys.stdout = log_file
    sys.stderr = log_file
    """
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    if model == "random":
        # use random mcts
        mcts_fn = _random_mcts_cpp
    else:
        # load torch model
        nn = torch.load("models/{}.pt".format(model)).cuda()
        mcts.evaluator = mcts.NNEvaluator(nn)
        mcts_fn = mcts.mcts
    memory_s = []
    memory_p = []
    memory_z = []
    points = np.arange(81)
    for ig in range(games_to_play):
        game_memory_s = []
        game_memory_p = []
        game_memory_z = []
        board_A = np.zeros((9, 9))
        board_B = np.zeros((9, 9))
        current_player = "A"
        res = 0
        ir = 0
        while True:
            actions_A = nogo.get_legal_actions(board_A, board_B)
            actions_B = nogo.get_legal_actions(board_B, board_A)
            aa = np.zeros((9, 9))
            ab = np.zeros((9, 9))
            for x, y in actions_A:
                aa[x][y] = 1
            for x, y in actions_B:
                ab[x][y] = 1
            if current_player == "A":
                p_map = mcts_fn(board_A, board_B, 400)
                game_memory_p.append(p_map)
                game_memory_s.append(
                    [deepcopy(board_A), deepcopy(board_B), aa, ab])
            else:
                p_map = mcts_fn(board_B, board_A, 400)
                game_memory_p.append(p_map)
                game_memory_s.append(
                    [deepcopy(board_B), deepcopy(board_A), aa, ab])
            num_actions_A = len(actions_A)
            num_actions_B = len(actions_B)
            if num_actions_A == 0 or num_actions_B == 0:
                if current_player == "A":
                    if num_actions_A == 0:
                        z = -1
                    else:
                        z = 1
                else:
                    if num_actions_B == 0:
                        z = 1
                    else:
                        z = -1
                print("seed:{} game:{} result:{} rounds:{}".format(
                    rand_seed, ig, z, len(game_memory_s)))
                for i in range(len(game_memory_s)):
                    game_memory_z.append(z)
                    z = -z
                break
            p_map = np.reshape(p_map, (81))
            p_map = p_map**(1/tempreature)
            p_map /= p_map.sum()
            action = np.random.choice(points, 1, p=p_map)[0]
            x, y = action//9, action % 9
            if current_player == "A":
                board_A[x][y] = 1
                current_player = "B"
            else:
                board_B[x][y] = 1
                current_player = "A"
            ir += 1
        # 这边需要去掉结束时的盘面，这个盘面不需要神经网络来判断
        memory_s += game_memory_s[:-1]
        memory_p += game_memory_p[:-1]
        memory_z += game_memory_z[:-1]
    memory_s = torch.tensor(np.array(memory_s, dtype=np.uint8))
    memory_p = torch.tensor(np.array(memory_p, dtype=np.int32))
    memory_z = torch.tensor(np.array(memory_z, dtype=np.int8))
    mem = {"s": memory_s, "p": memory_p, "z": memory_z}
    try:
        os.makedirs("self_play/{}".format(model))
    except:
        pass
    torch.save(mem,
               "self_play/{}/{}games_{}.pth".format(model, games_to_play, rand_seed))
