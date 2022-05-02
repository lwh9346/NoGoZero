#define bool char
#define false 0
#define true 1

// LegalActionResult是获取可落子位置的返回结果，应传入一个指针给该函数以便该函数将数据写回
struct LegalActionResult {
    int numS;      //己方可落子数
    int numR;      //对方可落子数
    char resS[81]; //己方可落子位置，x*9+y
    char resR[81]; //对方可落子位置，x*9+y
};

//找气的时候的分组
struct Group {
    bool self;           //是否为当前方
    int posCount;        //该组含有的子数
    int groupID;         //组别
    int libertyNum;      //拥有的气的数量
    char libertyPos[81]; //表明某个点是否为气，不是坐标
    char pos[81];        //该组所有子的坐标
};

struct Liberty {
    int ownerID[4];           //该气的所有组的组别
    int ownerLibertyCount[4]; //对应的组有多少个气
    int ownerCount;           //组数
    bool onlyLibertyS;        //己方唯一气
    bool onlyLibertyR;        //对方唯一气
    bool notOnlyLibertyS;     //己方非唯一气
    bool notOnlyLibertyR;     //对方非唯一气
};

void _grouping(int pos, int groupID, signed char board[81], signed char groupMark[81]) {
    groupMark[pos] = groupID;
    int x = pos / 9, y = pos % 9;
    int t = board[pos];
    int up = pos - 9;
    int down = pos + 9;
    int left = pos - 1;
    int right = pos + 1;
    if (x != 0 && groupMark[up] == -1 && t == board[up]) {
        _grouping(up, groupID, board, groupMark);
    }
    if (x != 8 && groupMark[down] == -1 && t == board[down]) {
        _grouping(down, groupID, board, groupMark);
    }
    if (y != 0 && groupMark[left] == -1 && t == board[left]) {
        _grouping(left, groupID, board, groupMark);
    }
    if (y != 8 && groupMark[right] == -1 && t == board[right]) {
        _grouping(right, groupID, board, groupMark);
    }
}

// get_legal_actions调用前要新建一个返回值结构体并将指针传进来
void get_legal_actions(signed char board[81], struct LegalActionResult *res) {
    //变量初始化
    signed char groupMark[81];
    struct Group groups[81];
    struct Liberty liberty[81];
    int groupNum = 0;
    res->numR = 0;
    res->numS = 0;
    for (int i = 0; i < 81; i++) {
        groupMark[i] = -1;
        liberty[i].ownerCount = 0;
        liberty[i].notOnlyLibertyR = false;
        liberty[i].notOnlyLibertyS = false;
        liberty[i].onlyLibertyR = false;
        liberty[i].onlyLibertyS = false;
        groups[i].groupID = i;
        groups[i].posCount = 0;
        groups[i].libertyNum = 0;
        for (int j = 0; j < 81; j++) {
            groups[i].libertyPos[j] = 0;
        }
    }

    //分组
    for (int i = 0; i < 81; i++) {
        if (groupMark[i] != -1 || board[i] == 0) { //跳过已分组元素与空格
            continue;
        }
        _grouping(i, groupNum, board, groupMark);
        groupNum++;
    }

    //填充分组结构体
    for (int i = 0; i < 81; i++) {
        if (groupMark[i] == -1) {
            continue;
        }
        groups[groupMark[i]].pos[groups[groupMark[i]].posCount] = i;
        groups[groupMark[i]].posCount++;
    }

    //找气
    for (int i = 0; i < groupNum; i++) {
        groups[i].self = board[(int)groups[i].pos[0]] == 1; //正反双方的判断放在这了
        for (int j = 0; j < groups[i].posCount; j++) {      //标记气
            int pos = groups[i].pos[j];
            int x = pos / 9, y = pos % 9;
            int up = pos - 9;
            int down = pos + 9;
            int left = pos - 1;
            int right = pos + 1;
            if (x != 0 && board[up] == 0) {
                groups[i].libertyPos[up] = 1;
            }
            if (x != 8 && board[down] == 0) {
                groups[i].libertyPos[down] = 1;
            }
            if (y != 0 && board[left] == 0) {
                groups[i].libertyPos[left] = 1;
            }
            if (y != 8 && board[right] == 0) {
                groups[i].libertyPos[right] = 1;
            }
        }
        int libertyPos[81];
        int libertyCount = 0;
        for (int j = 0; j < 81; j++) { //气数统计
            if (groups[i].libertyPos[j] == 1) {
                libertyPos[libertyCount] = j;
                libertyCount++;
            }
        }
        for (int j = 0; j < libertyCount; j++) { //填充气结构体
            liberty[libertyPos[j]].ownerLibertyCount[liberty[libertyPos[j]].ownerCount] = libertyCount;
            liberty[libertyPos[j]].ownerID[liberty[libertyPos[j]].ownerCount] = i;
            liberty[libertyPos[j]].ownerCount++;
        }
    }

    //处理气数据
    for (int i = 0; i < 81; i++) {
        for (int j = 0; j < liberty[i].ownerCount; j++) {
            if (groups[liberty[i].ownerID[j]].self) { //己方
                if (liberty[i].ownerLibertyCount[j] == 1) {
                    liberty[i].onlyLibertyS = true;
                    continue;
                }
                liberty[i].notOnlyLibertyS = true;
            } else { //对方
                if (liberty[i].ownerLibertyCount[j] == 1) {
                    liberty[i].onlyLibertyR = true;
                    continue;
                }
                liberty[i].notOnlyLibertyR = true;
            }
        }
    }

    //获取最终结果
    for (int pos = 0; pos < 81; pos++) {
        int x = pos / 9, y = pos % 9;
        int up = pos - 9;
        int down = pos + 9;
        int left = pos - 1;
        int right = pos + 1;
        bool atTop = x == 0;
        bool atBut = x == 8;
        bool atLef = y == 0;
        bool atRig = y == 8;

        //己方结果
        if (board[pos] != 0 || liberty[pos].onlyLibertyR) { //非空格或对方唯一气，不能下
            goto r;
        }
        if (liberty[pos].notOnlyLibertyS) { //己方非唯一气，能下
            res->resS[res->numS++] = pos;
            goto r;
        }
        if (liberty[pos].onlyLibertyS) { //如果是己方唯一气，还需判断
            //如果边上还有气就可以下
            if ((!atTop && board[up] == 0) || (!atBut && board[down] == 0) || (!atLef && board[left] == 0) || (!atRig && board[right] == 0)) {
                res->resS[res->numS++] = pos;
                goto r;
            }
            //边上没气就下不了了
            goto r;
        }
        if ((atTop || board[up] == -1) && (atBut || board[down] == -1) && (atLef || board[left] == -1) && (atRig || board[right] == -1)) { //死点
            goto r;
        }
        //啥都不是
        res->resS[res->numS++] = pos;

    r:
        //对方结果
        if (board[pos] != 0 || liberty[pos].onlyLibertyS) { //非空格或对方唯一气，不能下
            continue;
        }
        if (liberty[pos].notOnlyLibertyR) { //己方非唯一气，能下
            res->resR[res->numR++] = pos;
            continue;
        }
        if (liberty[pos].onlyLibertyR) { //如果是己方唯一气，还需判断
            //如果边上还有气就可以下
            if ((!atTop && board[up] == 0) || (!atBut && board[down] == 0) || (!atLef && board[left] == 0) || (!atRig && board[right] == 0)) {
                res->resR[res->numR++] = pos;
                continue;
            }
            //边上没气就下不了了
            continue;
        }
        if ((atTop || board[up] == 1) && (atBut || board[down] == 1) && (atLef || board[left] == 1) && (atRig || board[right] == 1)) { //死点
            continue;
        }
        //啥都不是
        res->resR[res->numR++] = pos;
    }
}
