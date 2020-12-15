import time
import sys
import argparse
import os
import numpy as np


def calculate_IC(graph, activity):
    count = len(activity)
    while len(activity):
        newac = []
        for i in activity:
            j = graph[i]
            c = 1
            while len(j) > c:
                if not graph[j[c][0]][0]:
                    if np.random.random() < j[c][1]:
                        graph[j[c][0]][0] = True
                        newac.append(j[c][0])
                c += 1
        count += len(newac)
        activity = newac
    return count


def each_core_IC(graph, activity, time_limit, start_time, time1):
    count = 0
    val = []
    while True:
        val.append(calculate_IC(graph, activity))
        count += 1
        now = time.time()
        if time_limit + start_time - now - 1 < (now - time1) / count:
            # print(count)
            break
        for i in range(len(graph)):
            graph[i][0] = False
        for e in activity:
            graph[e][0] = True
    return np.mean(val)


def calculate_LT(graph, activity):
    count = len(activity)
    while len(activity):
        newac = []
        for i in activity:
            j = graph[i]
            c = 3
            while len(j) > c:
                if not graph[j[c][0]][0]:
                    graph[j[c][0]][2] += j[c][1]
                    if graph[j[c][0]][1] <= graph[j[c][0]][2]:
                        graph[j[c][0]][0] = True
                        newac.append(j[c][0])
                c += 1
        count += len(newac)
        activity = newac
    return count


def each_core_LT(graph, activity, time_limit, start_time, time1):
    count = 0
    val = []
    while True:
        val.append(calculate_LT(graph, activity))
        count += 1
        now = time.time()
        if time_limit + start_time - now - 1 < (now - time1) / count:
            break
        for i in range(len(graph)):
            graph[i][0] = False
            graph[i][1] = np.random.random()
            graph[i][2] = 0
        for e in activity:
            graph[e][0] = True
    return np.mean(val)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='NetHEPT.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=120)
    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit

    activity = []
    seeds = open(seed, "r", encoding='utf-8').read()
    for e in seeds.split('\n'):
        activity.append(int(e))
    file = open(file_name)
    first = file.readline()
    head = first.split(' ')
    node = int(head[0])
    edge = int(head[1].split('\n')[0])
    if model == 'IC':
        graph = [[False] for i in range(node + 1)]
    else:
        graph = [[False, np.random.random(), 0] for i in range(node + 1)]
    for e in activity:
        graph[e][0] = True
    inp = np.genfromtxt(file_name, dtype=[int, int, float], skip_header=1)
    for i in range(edge):
        graph[inp[i][0]].append([inp[i][1], inp[i][2]])

    time1 = time.time()
    if model == 'IC':
        print(each_core_IC(graph, activity, time_limit, start_time, time1))
    else:
        print(each_core_LT(graph, activity, time_limit, start_time, time1))
    sys.stdout.flush()
    os._exit(0)
