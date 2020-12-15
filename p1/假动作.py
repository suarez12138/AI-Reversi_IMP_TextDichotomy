import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
Alpha = -1000
Beta = 1000
Topn = 6
Topnend = 5


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def calWendingziv2(self, chessboard, actcolor):
        point = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == actcolor:
                    vec = [(0, 1), (1, 0), (1, 1), (-1, 1)]
                    judge = [True, True, True, True]
                    for k in range(len(vec)):
                        tmp = (i + vec[k][0], j + vec[k][1])
                        while -1 < tmp[0] < 8 and -1 < tmp[
                            1] < 8:
                            if chessboard[tmp] == actcolor:
                                tmp = (tmp[0] + vec[k][0], tmp[1] + vec[k][1])
                            else:
                                judge[k] = False
                                break

                        if not judge[k]:
                            tmp = (i - vec[k][0], j - vec[k][1])
                            judge[k] = True
                            while -1 < tmp[0] < 8 and -1 < tmp[
                                1] < 8:
                                if chessboard[tmp] == actcolor:
                                    tmp = (tmp[0] - vec[k][0], tmp[1] - vec[k][1])
                                else:
                                    judge[k] = False
                                    break
                    if judge[0] and judge[1] and judge[2] and judge[3]:
                        point += 2

                    if sum(abs(chessboard[i])) == 8 and sum(
                            abs(chessboard[:, j])) == 8:
                        xie1 = True
                        xie2 = True
                        if i < j:
                            for k in range(8 - (j - i)):
                                if chessboard[k][j - i + k] == 0:
                                    xie1 = False
                                    break
                        else:
                            for k in range(8 - (i - j)):
                                if chessboard[j - i + k][k] == 0:
                                    xie1 = False
                                    break
                        if xie1:
                            if i + j <= 7:
                                for k in range(i + j + 1):
                                    if chessboard[i + j - k][k] == 0:
                                        xie2 = False
                                        break
                            else:
                                for k in range(abs(i - j) + 1):
                                    if chessboard[i + j - 7 + k][7 - k] == 0:
                                        xie2 = False
                                        break
                            if xie2:
                                point += 2
        return point

    def calWendingzi(self, chessboard, actcolor):
        point = 0
        border = [0, 8 - 1]
        for j in border:
            t1, t2 = False, False
            for i in range(6):
                if chessboard[j][i] == 0:
                    if t1 or t2:
                        break
                elif chessboard[j][i] == actcolor:
                    if t2:
                        point -= 20
                        break
                    t1 = True
                elif t1:
                    t2 = True
                else:
                    break
            t1, t2 = False, False
            for i in range(6):
                if chessboard[i][j] == 0:
                    if t1 or t2:
                        break
                elif chessboard[i][j] == actcolor:
                    if t2:
                        point -= 20
                        break
                    t1 = True
                elif t1:
                    t2 = True
                else:
                    break

        for j in border:
            t1, t2 = False, False
            for i in range(6):
                if chessboard[j][i] == 0:
                    if t1 or t2:
                        break
                elif chessboard[j][i] == -actcolor:
                    if t2:
                        point += 20
                        break
                    t1 = True
                elif t1:
                    t2 = True
                else:
                    break
            t1, t2 = False, False
            for i in range(6):
                if chessboard[i][j] == 0:
                    if t1 or t2:
                        break
                elif chessboard[i][j] == -actcolor:
                    if t2:
                        point += 20
                        break
                    t1 = True
                elif t1:
                    t2 = True
                else:
                    break

        if chessboard[0][0] == actcolor:
            if chessboard[0][1] == 0 and chessboard[0][7] == 0 and sum(chessboard[0]) == actcolor * 6:
                point -= 50
            if chessboard[1][0] == 0 and chessboard[7][0] == 0 and sum(chessboard[:, 0]) == actcolor * 6:
                point -= 50
        if chessboard[0][7] == actcolor:
            if chessboard[0][6] == 0 and chessboard[0][0] == 0 and sum(chessboard[0]) == actcolor * 6:
                point -= 50
            if chessboard[1][7] == 0 and chessboard[7][7] == 0 and sum(chessboard[:, 7]) == actcolor * 6:
                point -= 50
        if chessboard[7][7] == actcolor:
            if chessboard[7][6] == 0 and chessboard[7][0] == 0 and sum(chessboard[7]) == actcolor * 6:
                point -= 50
            if chessboard[6][7] == 0 and chessboard[0][7] == 0 and sum(chessboard[:, 7]) == actcolor * 6:
                point -= 50
        if chessboard[7][0] == actcolor:
            if chessboard[7][1] == 0 and chessboard[7][7] == 0 and sum(chessboard[7]) == actcolor * 6:
                point -= 50
            if chessboard[6][7] == 0 and chessboard[0][0] == 0 and sum(chessboard[:, 0]) == actcolor * 6:
                point -= 50
        return point

    def odd(self, chessboard):
        point = 0
        zero = [0, 0, 0, 0]
        down1 = [0, 0, 5, 5]
        up1 = [4, 4, 8, 8]
        down2 = [0, 5, 0, 5]
        up2 = [4, 8, 4, 8]
        for k in range(4):
            for i in range(down1[k], up1[k]):
                for j in range(down2[k], up2[k]):
                    if chessboard[i][j] == 0:
                        zero[k] += 1
            if zero[k] < 7:
                if zero[k] % 2 == 1:
                    point -= 20
        return point

    def xie(self, chessboard, actcolor):
        point = 0
        for i in range(2, 6):
            if chessboard[i][i] == actcolor:
                point += 20
        for i in range(2, 6):
            if chessboard[i][8 - i] == actcolor:
                point += 20
        return point

    def border(self, chessboard, actcolor):
        point = 0
        for i in range(8):
            for j in range(8):
                vec = [(0, 1), (1, 0), (1, 1), (-1, 1)]
                if chessboard[i][j] == actcolor:
                    for k in vec:
                        if -1 < i + k[0] < 8 and -1 < j + k[1] < 8 and chessboard[i + k[0]][j + k[1]] == 0:
                            point -= 2
                        if -1 < i - k[0] < 8 and -1 < j - k[1] < 8 and chessboard[i - k[0]][j - k[1]] == 0:
                            point -= 2
                elif chessboard[i][j] == -actcolor:
                    for k in vec:
                        if -1 < i + k[0] < 8 and -1 < j + k[1] < 8 and chessboard[i + k[0]][j + k[1]] == 0:
                            point += 2
                        if -1 < i - k[0] < 8 and -1 < j - k[1] < 8 and chessboard[i - k[0]][j - k[1]] == 0:
                            point += 2
        return point

    def evaluate(self, validMove, chessboard, actcolor):
        return sum(sum(chessboard * point)) * actcolor

    def placeable(self, chessboard, actcolor):
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        validBoard = []
        validMove = []
        vec = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        for i in idx:
            tmpBoard = chessboard.copy()
            valid = False
            for j in vec:
                tmp = (i[0] + j[0], i[1] + j[1])
                while -1 < tmp[0] < 8 and -1 < tmp[
                    1] < 8 and chessboard[tmp] == -actcolor:
                    tmp = (tmp[0] + j[0], tmp[1] + j[1])
                if -1 < tmp[0] < 8 and -1 < tmp[
                    1] < 8 and chessboard[tmp] == actcolor:
                    while True:
                        tmp = (tmp[0] - j[0], tmp[1] - j[1])
                        if tmp[0] == i[0] and tmp[1] == i[1]:
                            break
                        tmpBoard[tmp] = actcolor
                        valid = True
            if valid:
                validMove.append(i)
                tmpBoard[i] = actcolor
                validBoard.append(tmpBoard)
        return validBoard, validMove

    def pruning(self, chessboard, depth, maxdepth, alpha, beta, actcolor, end):
        validBoard, validMove = self.placeable(chessboard, actcolor)
        if len(validMove) == 0:
            return self.evaluate(validMove, chessboard, actcolor), (-1, -1)  #
        if depth == maxdepth:
            return self.evaluate(validMove, chessboard, actcolor), []

        if depth == 0:
            for i in validMove:
                if point[i[0]][i[1]] == point[0][0]:
                    return 100, i

        if depth <= maxdepth - 4:
            Values = []
            for i in range(len(validBoard)):
                value, bestMove = self.pruning(validBoard[i], maxdepth - 1, maxdepth, Alpha, Beta, -actcolor, end)
                Values.append(value)
                if bestMove == (-1, -1) and self.color == actcolor:
                    return 0, validMove[i]

            bestMoves = np.argsort(Values)  # 从大到小排序并按下标返回
            validMove2 = []
            validBoard2 = []
            top = Topn
            if end:
                top = Topnend
            for i in bestMoves[0:top]:
                validMove2.append(validMove[i])
                validBoard2.append(validBoard[i])
            validBoard = validBoard2
            validMove = validMove2

        bestMove = []
        bestScore = Alpha
        for i in range(len(validBoard)):
            score, useless = self.pruning(validBoard[i], depth + 1, maxdepth, -beta, -max(alpha, bestScore),
                                          -actcolor, end)
            score = -score
            if score > bestScore:
                bestMove = validMove[i]
                bestScore = score
                if bestScore > beta:
                    return bestScore, bestMove
        return bestScore, bestMove

    def go(self, chessboard):
        self.candidate_list.clear()
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        vec = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        for i in idx:
            for j in vec:
                valid = False
                inverse = False
                tmp = (i[0] + j[0], i[1] + j[1])
                while True:
                    if -1 < tmp[0] < 8 and -1 < tmp[1] < 8:
                        if chessboard[tmp] == -self.color:
                            inverse = True
                            tmp = (tmp[0] + j[0], tmp[1] + j[1])
                        elif inverse and chessboard[tmp] == self.color:
                            valid = True
                            self.candidate_list.append(i)
                            break
                        else:
                            break
                    else:
                        break
                if valid:
                    break
        end = False
        score, move = self.pruning(chessboard, 0, 2, Alpha, Beta, self.color, end)
        if move != (-1, -1):
            self.candidate_list.append(move)
        if move != (-1, -1):
            self.candidate_list.append(move)
            if (move == (1, 7) or (0, 6) or (1, 6) and chessboard[0][7] == 0) or (
                    move == (1, 0) or (0, 1) or (1, 1) and chessboard[0][0] == 0) or (
                    move == (7, 1) or (6, 0) or (6, 1) and chessboard[7][0] == 0) or (
                    move == (7, 6) or (6, 7) or (6, 6) and chessboard[7][7] == 0):
                self.candidate_list.pop()
        print(self.candidate_list)


point = np.array([
    [60, -5, 13, 9, 9, 13, -5, 60],
    [-5, -60, -6, 0, 0, -6, -60, -5],
    [13, -6, 2, 2, 2, 2, -6, 13],
    [9, 0, 2, -4, -4, 2, 0, 9],
    [9, 0, 2, -4, -4, 2, 0, 9],
    [13, -6, 2, 2, 2, 2, -6, 13],
    [-5, -60, -6, 0, 0, -6, -60, -5],
    [60, -5, 13, 9, 9, 13, -5, 60]
])
