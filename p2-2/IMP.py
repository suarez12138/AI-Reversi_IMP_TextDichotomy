import math
import random
import time
import sys
import argparse
import os
import numpy as np

# import multiprocessing as mp

# from p2t import ISE


def lb(n, k):
    f = 0
    for e in range(n - k + 1, n + 1):
        f += math.log(e)
    for e in range(1, k + 1):
        f -= math.log(e)
    return f


def Sampling(seedCount, epsilon, l):
    R = []
    LB = 1
    next_epsilon = epsilon * math.sqrt(2)
    log_bi = lb(node, seedCount)
    new_lambda = ((2 + 2 * next_epsilon / 3) * (
            log_bi + l * math.log(node) + math.log(math.log2(node))) * node) / pow(next_epsilon, 2)
    if model == 'IC':
        for i in range(1, int(math.log2(node))):
            x = node / math.pow(2, i)
            theta_i = new_lambda / x
            while len(R) <= theta_i:
                v = random.randint(1, node)
                RR = ICRR(v)
                R.append(RR)
            S_i, F_r = NodeSelection(R, seedCount)
            if node * F_r >= (1 + next_epsilon) * x:
                LB = node * F_r / (1 + next_epsilon)
                break
    else:
        for i in range(1, int(math.log2(node))):
            x = node / math.pow(2, i)
            theta_i = new_lambda / x
            while len(R) <= theta_i:
                v = random.randint(1, node)
                RR = LTRR(v)
                R.append(RR)
            S_i, F_r = NodeSelection(R, seedCount)
            if node * F_r >= (1 + next_epsilon) * x:
                LB = node * F_r / (1 + next_epsilon)
                break

    alpha = math.sqrt(l * math.log(node) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (log_bi + l * math.log(node) + math.log(2)))
    lambda_star = 2 * node * math.pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsilon, -2)
    theta = lambda_star / LB

    if model == 'IC':
        while len(R) <= theta:
            n = random.randint(1, node)
            R.append(ICRR(n))
    else:
        while len(R) <= theta:
            n = random.randint(1, node)
            R.append(LTRR(n))
    return R


def ICRR(n):
    RR = set()
    RR.add(n)
    ac = set()
    ac.add(n)
    while ac:
        new_ac = set()
        for e in ac:
            j = reverse_graph[e]
            for c in range(0, len(j)):
                if j[c][0] not in RR and random.random() < j[c][1]:
                    RR.add(j[c][0])
                    new_ac.add(j[c][0])
        ac = new_ac
    return RR


def LTRR(n):
    RR = set()
    RR.add(n)
    ac = n
    while ac != -1:
        n_ac = -1
        if reverse_graph[ac]:
            r = random.randint(0, len(reverse_graph[ac]) - 1)
            if reverse_graph[ac][r] not in RR:
                RR.add(reverse_graph[ac][r])
                n_ac = reverse_graph[ac][r]
        ac = n_ac
    return RR


def NodeSelection(R, seedCount):
    S_k = set()
    RR_correspoding = {}
    node_count = [0 for i in range(node + 1)]
    for j in range(len(R)):
        for each_node in R[j]:
            node_count[each_node] += 1
            if each_node not in RR_correspoding:
                RR_correspoding[each_node] = []
            RR_correspoding[each_node].append(j)
    for i in range(seedCount):
        max_apper = node_count.index(max(node_count))
        if max_apper == 0:
            z = i
            pos = 1
            for i in range(z, seedCount):
                for x in range(pos, node + 1):
                    if x not in S_k:
                        S_k.add(x)
                        break
            break
        S_k.add(max_apper)
        having_max = []
        for each_node in RR_correspoding[max_apper]:
            having_max.append(each_node)
        for R_index in having_max:
            for p in R[R_index]:
                node_count[p] -= 1
                RR_correspoding[p].remove(R_index)
    count = 0
    for RR in R:
        for s in S_k:
            if s in RR:
                count += 1
    return list(S_k), count / len(R)


# def trying():
#     R = []
#     time1 = time.time()
#     if model == 'IC':
#         while time1 - start_time < time_limit - 55:
#             v = random.randint(1, node)
#             RR = ICRR(v)
#             R.append(RR)
#             time1 = time.time()
#     print(len(R))
#     return R


def imm(seedCount, epsilon, l):
    l = l * (1 + math.log(2) / math.log(node))
    # timek = time.time()
    R = Sampling(seedCount, epsilon, l)
    # R = trying()
    # time2 = time.time()
    # print(time2 - timek)
    S_k, nouse = NodeSelection(R, seedCount)
    # print(time.time() - time2)
    return S_k


if __name__ == '__main__':
    global seedCount, model, node, reverse_graph, time_limit, start_time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='test.txt')
    parser.add_argument('-k', '--seedCount', type=int, default='1000')
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=120)
    args = parser.parse_args()
    file_name = args.file_name
    seedCount = args.seedCount
    model = args.model
    time_limit = args.time_limit

    file = open(file_name)
    first = file.readline()
    head = first.split(' ')
    node = int(head[0])
    edge = int(head[1].split('\n')[0])

    reverse_graph = [[] for i in range(node + 1)]

    inp = np.genfromtxt(file_name, dtype=[int, int, float], skip_header=1)
    if model == 'IC':
        for i in range(edge):
            reverse_graph[inp[i][1]].append([inp[i][0], inp[i][2]])
    else:
        for i in range(edge):
            reverse_graph[inp[i][1]].append(inp[i][0])

    epsilon = 0.1
    # if model == 'IC':
    #     if node < 1001:
    #         epsilon = 0.06
    #     elif node < 5001:
    #         epsilon = 0.08
    #     # elif node > 10000:
    #     #     epsilon = 0.08
    #     #     # epsilon = 0.18 # 跑lt seed 5
    #     elif node < 16000 and time_limit <= 60:
    #         epsilon = 0.07
    #     elif node < 16000 and time_limit >= 120 and seedCount <= 50:
    #         epsilon = 0.1
    #         # epsilon = 0.08
    #     elif node < 50001 or seedCount == 500:
    #         epsilon = 0.1
    # else:
    if node < 1001:
        epsilon = 0.06
    elif node < 5001:
        epsilon = 0.08
    elif node < 16000 and time_limit <= 60:
        epsilon = 0.1
    elif node < 16000 and time_limit >= 120:
        epsilon = 0.08
    elif node < 50001:
        epsilon = 0.14
    elif node > 60000:
        epsilon = 0.2

    result = imm(seedCount, epsilon, 1)

    for seed in result:
        print(seed)
    print(time.time()-start_time)
    # test = [0 for i in range(node + 1)]
    # for e in result:
    #     test[e] += 1
    # for i in range(node + 1):
    #     if test[i] > 1:
    #         print(str(i) + ':' + str(test[i]))

    # if model == 'IC':
    #     graph = [[False] for i in range(node + 1)]
    # else:
    #     graph = [[False, np.random.random(), 0] for i in range(node + 1)]
    # for e in result:
    #     graph[e][0] = True
    # inp = np.genfromtxt(file_name, dtype=[int, int, float], skip_header=1)
    # for i in range(edge):
    #     graph[inp[i][0]].append([inp[i][1], inp[i][2]])
    # time1 = time.time()
    # print(time1 - start_time)
    # if model == 'IC':
    #     print(ISE.each_core_IC(graph, result, 100, start_time, time1))
    # else:
    #     print(ISE.each_core_LT(graph, result, 150, start_time, time1))

    sys.stdout.flush()
    os._exit(0)
