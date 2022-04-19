#include "bot.cpp"
#include <cstring>
#include <ctime>
#include <iostream>
#include <math.h>
#include <random>
#include <string>
using namespace std;

int main() {
    //计数器与棋盘声明，随机数初始化
    signed char board[81] = {0};
    srand((unsigned)time(NULL));

    //读入数据
    string str;
    getline(cin, str);
    for (int i = 0; i < 81; i++) {
        switch (str[i]) {
        case 'A':
            board[i] = 1;
            break;
        case 'B':
            board[i] = -1;
            break;
        case 'C':
            board[i] = 0;
            break;
        default:
            return 1;
        }
    }
    mcts::DebugData debugData;
    mcts::GetBestAction(board, 500, &debugData);

    for (int i = 0; i < 81; i++) {
        printf("%d ", debugData.nMap[i]);
    }

    return 0;
}