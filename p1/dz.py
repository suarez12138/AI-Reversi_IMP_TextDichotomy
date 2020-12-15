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


# [64][2]
# hashValue = {
#     {4594751534691652102, 1851944401440995956}, {2747968243137263198, 2207742792711565793},
#     {3848004645121087451, 3216815717814836416}, {1516548209265803966, 2020642571208227878},
#     {2337722696409584694, 3437059740345402271}, {2898258925230917030, 1303855352451580365},
#     {6741448498992359, 4377852815973168715}, {4372139272928371625, 3315674304057407093},
#     {2878955577234187687, 3310911488601986908}, {3582136329572115185, 3886893032669033165},
#     {708434475528378106, 1888811197289616306}, {1100757220377106200, 2346419301461881836},
#     {1995470285667003037, 3344903240187633289}, {379138142664434743, 2911313461119289748},
#     {769900647706363980, 4570344310535739107}, {3830265281089289793, 3708897551051154464},
#     {3315091279816950970, 273872670825111973}, {2041102417123238912, 3112772639692668910},
#     {4299351587052741497, 492310524505137784}, {4010522851494399473, 362833556870927538},
#     {3744809366975419281, 4226902705789150687}, {592324386571251539, 1676350313352030418},
#     {297655183198752013, 4509827100803281127}, {1403475634077929604, 1892578936484963221},
#     {1715531292376832429, 1613769283609259830}, {504753725576921797, 2392422484281687273},
#     {1111776814178973127, 1894484583724300455}, {1353368917078917269, 980842078909767887},
#     {3122588916900066851, 1904920763832515654}, {604593662070056141, 4610439775266802671},
#     {3503933470494872029, 465452319321546006}, {4524482860969100773, 418483288981978408},
#     {2147108479613262724, 402603979465422751}, {1068073449811579883, 773275828138712383},
#     {313392534030852928, 2844281495097492616}, {4421182344573758470, 2690463909089054058},
#     {3564969729531244233, 2153253714589235195}, {2048902308496474304, 2035312525942884256},
#     {247110080496067937, 460126216295236977}, {2910862077868453624, 1236872844561888511},
#     {4434503285626937640, 1788235057956326105}, {617585277383179775, 3098146742097029318},
#     {544179776494490277, 3070740981001682629}, {1133217526157530591, 2597739258228835604},
#     {2291314423671849210, 3285761339244847317}, {3210152689877290283, 4256064628862687295},
#     {983114329860268105, 669680476452377688}, {3546030145041577463, 3617169695558622675},
#     {3355229619035115716, 4340260948930533822}, {4157463891367309266, 4259738129252256818},
#     {3868311963489900326, 292106685109972994}, {3029152897729491494, 443633747780305188},
#     {4438782935932979823, 476344706333946286}, {225228141352298397, 437312018862372058},
#     {3013545812034048651, 1882826361476336167}, {1682028769400164518, 490632862661383976},
#     {3045929461469737835, 705051332461977505}, {3071896532965914314, 4398387264278704174},
#     {2968653874300097863, 2681664983844070838}, {2243007804517107923, 2718365947100986101},
#     {3315433970204178064, 2094902054428670605}, {3639206564310825114, 495668310978194141},
#     {3266804706767133950, 137318566489138708}, {847159218020085283, 3988956296565724071},
# }


class Hashentry(object):
    def __init__(self, zobrist, depth, flag, eval, ancient, move):
        self.zobrist = zobrist


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    # def special(self, chessboard, actcolor):
    #     point = 0
    #     if chessboard[0][0] == 0:
    #         if chessboard[0][1] == actcolor and chessboard[0][2] == -actcolor or chessboard[1][0] == actcolor and \
    #                 chessboard[2][0] == -actcolor:
    #             point += -50
    #         if chessboard[1][1] == actcolor:
    #             point += -50
    #     if chessboard[0][self.chessboard_size - 1] == 0:
    #         if chessboard[0][6] == actcolor and chessboard[0][
    #             self.chessboard_size - 3] == -actcolor or chessboard[1][self.chessboard_size - 1] == actcolor and \
    #                 chessboard[2][self.chessboard_size - 1] == -actcolor:
    #             point += -50
    #         if chessboard[1][6] == actcolor:
    #             point += -50
    #     if chessboard[self.chessboard_size - 1][0] == 0:
    #         if chessboard[6][0] == actcolor and chessboard[
    #             self.chessboard_size - 3][0] == -actcolor or chessboard[self.chessboard_size - 1][1] == actcolor and \
    #                 chessboard[self.chessboard_size - 1][2] == -actcolor:
    #             point += -50
    #         if chessboard[6][1] == actcolor:
    #             point += -50
    #     if chessboard[self.chessboard_size - 1][self.chessboard_size - 1] == 0:
    #         if chessboard[6][self.chessboard_size - 1] == actcolor and chessboard[
    #             self.chessboard_size - 3][self.chessboard_size - 1] == -actcolor or \
    #                 chessboard[self.chessboard_size - 1][6] == actcolor and \
    #                 chessboard[self.chessboard_size - 1][self.chessboard_size - 3] == -actcolor:
    #             point += -50
    #         if chessboard[6][6] == actcolor:
    #             point += -50
    #     return point

    def calWendingziv2(self, chessboard, actcolor):
        point = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] == actcolor:
                    vec = [(0, 1), (1, 0), (1, 1), (-1, 1)]
                    judge = [True, True, True, True]
                    for k in range(len(vec)):
                        tmp = (i + vec[k][0], j + vec[k][1])
                        while -1 < tmp[0] < self.chessboard_size and -1 < tmp[
                            1] < self.chessboard_size:
                            if chessboard[tmp] == actcolor:
                                tmp = (tmp[0] + vec[k][0], tmp[1] + vec[k][1])
                            else:
                                judge[k] = False
                                break

                        if not judge[k]:
                            tmp = (i - vec[k][0], j - vec[k][1])
                            judge[k] = True
                            while -1 < tmp[0] < self.chessboard_size and -1 < tmp[
                                1] < self.chessboard_size:
                                if chessboard[tmp] == actcolor:
                                    tmp = (tmp[0] - vec[k][0], tmp[1] - vec[k][1])
                                else:
                                    judge[k] = False
                                    break
                    if judge[0] and judge[1] and judge[2] and judge[3]:
                        point += 2

                    if sum(abs(chessboard[i])) == self.chessboard_size and sum(
                            abs(chessboard[:, j])) == self.chessboard_size:
                        xie1 = True
                        xie2 = True
                        if i < j:
                            for k in range(self.chessboard_size - (j - i)):
                                if chessboard[k][j - i + k] == 0:
                                    xie1 = False
                                    break
                        else:
                            for k in range(self.chessboard_size - (i - j)):
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
        border = [0, self.chessboard_size - 1]
        for j in border:
            t1, t2 = False, False
            for i in range(6):
                if chessboard[j][i] == 0:
                    if t1 or t2:
                        break
                elif chessboard[j][i] == actcolor:
                    if t2:
                        point -= 30
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
                        point -= 30
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
                        point += 30
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
                        point += 30
                        break
                    t1 = True
                elif t1:
                    t2 = True
                else:
                    break

        if chessboard[0][0] == actcolor:
            if chessboard[0][1] == 0 and chessboard[0][7] == 0 and sum(chessboard[0]) == actcolor * 6:
                point -= 30
            if chessboard[1][0] == 0 and chessboard[7][0] == 0 and sum(chessboard[:, 0]) == actcolor * 6:
                point -= 30

        if chessboard[0][7] == actcolor:
            if chessboard[0][6] == 0 and chessboard[0][0] == 0 and sum(chessboard[0]) == actcolor * 6:
                point -= 30
            if chessboard[1][7] == 0 and chessboard[7][7] == 0 and sum(chessboard[:, 7]) == actcolor * 6:
                point -= 30
        if chessboard[7][7] == actcolor:
            if chessboard[7][6] == 0 and chessboard[7][0] == 0 and sum(chessboard[7]) == actcolor * 6:
                point -= 30
            if chessboard[6][7] == 0 and chessboard[0][7] == 0 and sum(chessboard[:, 7]) == actcolor * 6:
                point -= 30
        if chessboard[7][0] == actcolor:
            if chessboard[7][1] == 0 and chessboard[7][7] == 0 and sum(chessboard[7]) == actcolor * 6:
                point -= 30
            if chessboard[6][7] == 0 and chessboard[0][0] == 0 and sum(chessboard[:, 0]) == actcolor * 6:
                point -= 30
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
                point += 30
                break
        for i in range(2, 6):
            if chessboard[i][8 - i] == actcolor:
                point += 30
                break
        return point

    def evaluate(self, validMove, chessboard, actcolor):
        wendingzi2 = self.calWendingziv2(chessboard, actcolor)
        wendingzi = self.calWendingzi(chessboard, actcolor)
        # spe = self.special(chessboard, actcolor)
        validBoard, useless = self.placeable(chessboard, -actcolor)
        ifodd = self.odd(chessboard)
        xied = self.xie(chessboard, actcolor)

        return sum(sum(chessboard * point)) * actcolor + 6 * (
                len(validMove) - len(validBoard)) + wendingzi + wendingzi2 + ifodd + xied

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
                while -1 < tmp[0] < self.chessboard_size and -1 < tmp[
                    1] < self.chessboard_size and chessboard[tmp] == -actcolor:
                    tmp = (tmp[0] + j[0], tmp[1] + j[1])
                if -1 < tmp[0] < self.chessboard_size and -1 < tmp[
                    1] < self.chessboard_size and chessboard[tmp] == actcolor:
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
                    if -1 < tmp[0] < self.chessboard_size and -1 < tmp[1] < self.chessboard_size:
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
        place = sum(sum(abs(chessboard)))
        end = False
        if place < 10:
            depth = 5
        elif place > 54:
            depth = 9
        elif place > 53:
            depth = 7
        elif place > 47:
            depth = 6
            # end = True
        elif place > 45:
            depth = 5
            # end = True
        else:
            depth = 10
        t = time.time()
        score, move = self.pruning(chessboard, 0, 2, Alpha, Beta, self.color, end)
        if move != (-1, -1):
            self.candidate_list.append(move)
        print(time.time() - t)
        score, move = self.pruning(chessboard, 0, depth, Alpha, Beta, self.color, end)
        if move != (-1, -1):
            self.candidate_list.append(move)
        print(self.candidate_list)


point = np.array([
    [60, -5, 13, 9, 9, 13, -5, 60],
    [-5, -50, -6, 0, 0, -6, -50, -5],
    [13, -6, 2, 2, 2, 2, -6, 13],
    [9, 0, 2, -4, -4, 2, 0, 9],
    [9, 0, 2, -4, -4, 2, 0, 9],
    [13, -6, 2, 2, 2, 2, -6, 13],
    [-5, -50, -6, 0, 0, -6, -50, -5],
    [60, -5, 13, 9, 9, 13, -5, 60]
])

if __name__ == '__main__':
    # li = np.array([[-1, 1, -1, -1, -1, -1, 0, 0],
    #                [-1, 1, 1, 1, 1, 1, 0, 0],
    #                [-1, -1, -1, -1, 1, 1, 1, 0],
    #                [-1, -1, -1, -1, -1, 1, 1, -1],
    #                [-1, -1, -1, 1, -1, -1, 1, -1],
    #                [-1, -1, 1, -1, -1, -1, -1, -1],
    #                [0, 0, -1, -1, -1, -1, 0, -1],
    #                [0, 0, -1, -1, -1, -1, 0, 0]])
    li2 = np.array([[0, 0, -1, -1, -1, -1, -1, 0],
                    [-1, 0, -1, -1, -1, -1, 0, 0],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 1, 1, 1, 1, -1, -1],
                    [0, 0, -1, -1, -1, 1, 1, -1],
                    [0, 0, -1, -1, -1, 1, 1, -1],
                    [0, 0, -1, -1, -1, -1, 0, -1]])
    li4 = np.array([[0, 0, -1, -1, -1, -1, -1, 0],
                    [-1, 0, -1, -1, -1, -1, 0, 0],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 0, 1, 1, 1, 1, 1, -1],
                    [0, 0, -1, -1, -1, 1, 1, -1],
                    [0, 0, -1, -1, -1, -1, 0, -1]])
    # li3 = np.array(
    #     [[0, 0, 0, 1, 0, -1, 0, 0],
    #      [0, 0, 0, 0, 1, -1, 0, 0],
    #      [0, 0, 1, -1, -1, 1, 0, 0],
    #      [0, 0, 1, -1, 1, 1, 1, 0],
    #      [0, 0, 1, -1, 1, 0, 0, 0],
    #      [0, 0, 1, -1, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0]])

    li5 = np.array([[0, 0, -1, -1, -1, 0, 0, 0],
                    [0, 1, -1, -1, -1, 0, 0, -1],
                    [-1, 1, 1, -1, -1, 1, -1, -1],
                    [-1, 1, -1, 1, -1, -1, 1, -1],
                    [-1, -1, -1, -1, 1, 1, -1, 0],
                    [-1, 1, -1, 1, -1, 1, 1, 1],
                    [1, 1, 1, 1, -1, -1, 0, 0],
                    [-1, -1, -1, -1, -1, -1, -1, 0]])
    li6 = np.array([[0, -1, -1, -1, -1, -1, -1, 0],
                    [0, 0, -1, -1, 1, 1, 1, -1],
                    [0, -1, -1, -1, -1, 1, 1, -1],
                    [1, 1, 1, 1, 1, -1, 1, -1],
                    [0, 1, -1, 1, -1, 1, -1, -1],
                    [-1, -1, 1, -1, -1, -1, -1, -1],
                    [0, 1, -1, 1, 0, 0, 0, 0],
                    [0, 0, -1, -1, -1, -1, 0, 0]])
    li7 = np.array(
        [[0, 0, 0, 0, 0, 1, 0, 0],
         [-1, 0, 0, 0, 1, 0, 0, 0],
         [-1, 0, 1, 1, 1, 1, 0, 0],
         [-1, 1, 1, 1, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, -1, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 0, 0, 0]])
    li8 = np.array(
        [[0, 0, -1, -1, -1, -1, -1, -1],
         [0, 0, -1, -1, 1, -1, 1, -1],
         [0, -1, -1, -1, -1, -1, 1, -1],
         [0, -1, 1, 1, 1, -1, -1, -1],
         [-1, 0, 1, 1, -1, -1, -1, -1],
         [0, 1, 1, 1, -1, -1, -1, -1],
         [0, 0, 1, 1, 1, 1, 0, -1],
         [0, 0, -1, 0, 1, -1, 0, 0]]
    )
    li9 = np.array(
        [[0, 0, -1, -1, -1, -1, 0, 0],
         [0, 0, -1, -1, -1, -1, 0, 0],
         [1, 1, -1, 1, -1, -1, -1, -1],
         [1, 1, 1, -1, -1, -1, -1, -1],
         [-1, 1, 1, 1, 1, 1, 1, -1],
         [1, 1, 1, 1, 1, 1, 1, -1],
         [0, 0, 0, 0, -1, -1, 0, -1],
         [0, 0, 0, -1, -1, -1, 0, 0]]
    )
    li10 = np.array(
        [[0, 0, -1, 1, 0, -1, 0, 0],
         [-1, 0, 1, 1, 1, 0, 0, -1],
         [-1, 1, -1, 1, 1, 1, -1, -1],
         [-1, -1, -1, 1, 1, -1, -1, -1],
         [-1, -1, -1, 1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1, -1],
         [0, 0, 0, -1, -1, -1, 0, 0],
         [0, 0, 0, -1, -1, -1, 0, 0]]
    )
    li11 = np.array(
        [[0, 0, -1, 0, 1, 1, 1, 0],
         [0, 0, 0, -1, 1, -1, 0, 0],
         [0, 0, 1, 1, -1, -1, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, -1, -1, 1, 1, 0],
         [0, 0, -1, 1, -1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, -1, -1, -1, -1, -1, 0]]
    )
    li12 = np.array(
        [[0, 0, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, -1, -1, 0, 0], [1, 0, 1, -1, -1, -1, 0, -1],
         [1, -1, 1, 1, 1, -1, -1, -1], [0, 0, -1, 1, 1, -1, -1, -1], [0, -1, -1, -1, -1, -1, -1, -1],
         [0, 0, -1, 1, -1, -1, 0, -1], [0, -1, -1, -1, -1, -1, -1, 0]]
    )
    li13 = np.array([[0, 0, 1, 1, 1, 1, 1, 0],
                     [1, 0, -1, -1, -1, 1, 0, 0],
                     [1, 1, -1, -1, 1, 1, -1, -1],
                     [1, -1, -1, 1, 1, -1, -1, -1],
                     [1, 1, -1, -1, -1, 1, -1, -1],
                     [1, 1, 1, -1, -1, -1, -1, -1],
                     [1, 0, -1, -1, 0, 0, 0, 1],
                     [0, 0, -1, -1, 1, 0, 0, 0]])
    li14 = np.array([[0, -1, -1, -1, -1, -1, 0, 0],
                     [0, 0, -1, -1, -1, -1, 0, -1],
                     [-1, -1, -1, -1, -1, -1, -1, -1],
                     [0, -1, -1, -1, -1, -1, -1, -1],
                     [0, 1, -1, -1, 1, -1, -1, -1],
                     [1, 1, 1, 1, -1, -1, -1, -1],
                     [0, 0, 1, -1, -1, -1, 0, 0],
                     [0, 1, 1, 1, 1, 0, 0, 0]])
    li15 = np.array(
        [[0, 0, 1, 1, 1, 1, 1, 0],
         [0, -1, -1, -1, -1, -1, 0, 0],
         [0, 0, -1, -1, -1, -1, 0, -1],
         [1, 1, 1, -1, 1, -1, -1, -1],
         [0, 1, 1, 1, -1, -1, -1, -1],
         [0, -1, 1, -1, -1, -1, -1, -1],
         [0, 0, -1, 1, -1, -1, 0, -1],
         [0, -1, -1, -1, -1, -1, -1, 0]]
    )
    li16 = np.array(
        [[1, -1, -1, -1, -1, -1, -1, -1],
         [0, 1, -1, -1, 1, -1, 1, -1],
         [1, 1, 1, 1, -1, 1, 1, -1],
         [0, 1, 1, -1, 1, 1, 1, -1],
         [1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, -1, 1, 1, 1, -1], [1, 1, 1, 1, 1, 1, 1, -1],
         [1, 0, 1, 1, 1, -1, 0, -1]]
    )
    li17 = np.array(
        [[0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, -1, 1, 1, 0, 0], [1, 1, 1, 1, -1, 1, 1, 0], [1, -1, 1, -1, -1, -1, 1, 1],
         [1, 1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, -1, -1], [0, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, -1, -1, 1, -1, 0, 0]]
    )
    li18 = np.array(
        [[0, 0, -1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, -1], [1, 1, -1, 1, 1, 1, -1, -1], [1, 1, 1, 1, 1, -1, 1, -1],
         [1, 1, 1, 1, -1, -1, 1, -1], [0, 1, 1, -1, -1, -1, 1, -1], [0, 1, -1, 1, 1, 1, 0, -1],
         [0, -1, -1, 0, 1, 1, 1, 0]]
    )
    li19 = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, -1, -1, -1, -1, -1, 0],
         [0, 0, 0, -1, 1, -1, -1, -1], [1, 1, 0, -1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, 1, -1],
         [1, 0, 1, -1, -1, -1, 0, -1], [0, 1, 1, 1, 1, 1, 1, 0]]
    )
    a = time.time()
    ai = AI(8, COLOR_BLACK, 10)
    ai.go(li19)
    print(time.time() - a)
