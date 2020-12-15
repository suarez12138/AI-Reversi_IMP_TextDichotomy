import math
import random
import sys
import argparse
import os
import numpy as np


def lb(n, k):
    f = 0
    for e in range(n - k + 1, n + 1):
        f += math.log(e)
    for e in range(1, k + 1):
        f -= math.log(e)
    return f


def Sampling(epsilon, l):
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
            S_i = NodeSelection(R)
            count = 0
            for RR in R:
                for s in S_i:
                    if s in RR:
                        count += 1
            F_r = count / len(R)
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
            S_i = NodeSelection(R)
            count = 0
            for RR in R:
                for s in S_i:
                    if s in RR:
                        count += 1
            F_r = count / len(R)
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


def NodeSelection(R):
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
    return list(S_k)


def imm(epsilon, l):
    l = l * (1 + math.log(2) / math.log(node))
    R = Sampling(epsilon, l)
    S_k = NodeSelection(R)
    return S_k


if __name__ == '__main__':
    global seedCount, model, node, reverse_graph, start_time
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='NetHEPT.txt')
    parser.add_argument('-k', '--seedCount', type=int, default='50')
    parser.add_argument('-m', '--model', type=str, default='IC')
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

    if node / edge > 1 / 4:
        epsilon = 0.1
        if node < 1001:
            epsilon = 0.06
        elif node < 5001:
            epsilon = 0.08
        elif node < 16000 and time_limit <= 60:
            epsilon = 0.08
        elif node < 16000 and time_limit >= 120 and seedCount < 101:
            epsilon = 0.06
        elif node < 50001:
            epsilon = 0.14
        elif node > 60000:
            epsilon = 0.2
    elif node / edge > 1 / 6:
        epsilon = 0.1
        if node < 1001:
            epsilon = 0.09
        elif node < 5001:
            epsilon = 0.12
        elif node < 16000 and time_limit <= 60:
            epsilon = 0.2
        elif node < 16000 and time_limit >= 120 and seedCount < 101:
            epsilon = 0.15
        elif node < 50001:
            epsilon = 0.4
        elif node > 60000:
            epsilon = 0.6
    else:
        epsilon = 0.1
        if node < 501:
            epsilon = 0.08
        elif node < 1001:
            epsilon = 0.1
        elif node < 5001:
            epsilon = 0.15
        elif node < 16000 and time_limit <= 60:
            epsilon = 0.25
        elif node < 16000 and time_limit >= 120 and seedCount < 101:
            epsilon = 0.2
        elif node < 50001:
            epsilon = 0.6
        elif node > 60000:
            epsilon = 0.8

    result = imm(epsilon, 1)

    for seed in result:
        print(seed)
    sys.stdout.flush()
    os._exit(0)
