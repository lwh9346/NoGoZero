import numpy as np
import torch
from data_generator import random_mcts_cpp
import mcts
import nogo


def pure_random(board_A, board_B, max_N):
    res = np.zeros((9, 9))
    for x, y in nogo.get_legal_actions(board_A, board_B):
        res[x][y] = 1
    return res


if __name__ == "__main__":
    # 设置
    mcts.evaluator = mcts.NNEvaluator(torch.load("models/model_1.pt"))
    RANDOM_MODEL = random_mcts_cpp
    DEEP_LEARNING_MODEL = mcts.mcts
    TEMPREATURE = 0.5
    GAMES_TO_PLAY = 100
    SEARCH_STEPS = 500
    # 变量初始化
    model_dl_wins = 0
    for i in range(GAMES_TO_PLAY):
        mcts_fn_A = DEEP_LEARNING_MODEL if i % 2 == 0 else RANDOM_MODEL
        mcts_fn_B = DEEP_LEARNING_MODEL if i % 2 == 1 else RANDOM_MODEL
        board_A, board_B = np.zeros((9, 9)), np.zeros((9, 9))
        while True:
            # 棋局状态判断
            action_A = nogo.get_legal_actions(board_A, board_B)
            action_B = nogo.get_legal_actions(board_B, board_A)
            if len(action_A) == 0 or len(action_B) == 0:
                # 终局
                if (len(action_A) == 0 and mcts_fn_B is DEEP_LEARNING_MODEL) or (len(action_B) == 0 and mcts_fn_A is DEEP_LEARNING_MODEL):
                    model_dl_wins += 1
                    print("对局{} 胜者:dl".format(i))
                else:
                    print("对局{} 胜者:rand".format(i))
                break
            # 落子
            p = mcts_fn_A(board_A, board_B, SEARCH_STEPS)
            p = np.reshape(p, (81))
            if TEMPREATURE != 0:
                p = p**(1/TEMPREATURE)
                p /= np.sum(p)
                a = np.random.choice(81, p=p)
            else:
                a = np.argmax(p)
            x, y = a//9, a % 9
            board_A[x][y] = 1
            # 交换盘面
            mcts_fn_A, mcts_fn_B = mcts_fn_B, mcts_fn_A
            board_A, board_B = board_B, board_A
    print("总计游戏:{} dl获胜:{} 胜率:{:.1f}%".format(GAMES_TO_PLAY,
                                              model_dl_wins, 100*model_dl_wins/GAMES_TO_PLAY))
