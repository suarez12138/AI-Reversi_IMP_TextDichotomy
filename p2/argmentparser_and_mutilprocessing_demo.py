# -*- coding: utf-8 -*-
# written by mark zeng 2018-11-14
# modified by Yao Zhao 2019-10-30
# re-modified by Yiming Chen 2020-11-04

import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np

core = 8


def sum_and_product(x, y):
    '''
    计算两个数的和与积
    '''
    while True:
        x = x + y
    return x + y, x * y


if __name__ == '__main__':
    '''
    从命令行读参数示例
    '''
    print("从命令行读参数示例")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit

    print(file_name, seed, model, time_limit)

    '''
    多进程示例
    '''
    print("多进程示例")
    np.random.seed(0)
    pool = mp.Pool(core)
    result = []

    for i in range(core):
        result.append(pool.apply_async(sum_and_product, args=(np.random.randint(0, 10), np.random.randint(0, 10))))
    pool.close()
    pool.join()

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
