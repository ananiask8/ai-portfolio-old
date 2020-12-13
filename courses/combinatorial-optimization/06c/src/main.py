#!/usr/bin/env python2

import sys
from collections import defaultdict
from random import randint, random
from numpy.random import choice
import copy
from time import time
from math import exp
starting_time = time()

# Parameters
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")
time_limit = int(sys.argv[3])

s = input_file.readline().strip()
p = defaultdict(lambda : defaultdict(dict))
h = 10000
V = set()
for line in input_file.readlines():
    ci, cj, d, pijd = line.split(" ")
    p[int(d)][ci][cj] = int(pijd) + h # make metric
    V.add(ci)
input_file.close()

def getHC(p, V, s):
    N = len(V)
    v = [s]
    total = 0
    V.remove(s)
    possible_destinations = V.intersection(p[0][v[-1]].keys())
    for i in range(N - 1):
        updated_map = {k: v for k, v in p[i][v[-1]].items() if k in possible_destinations}
        c = min(updated_map.values())
        total += c
        v.append([w for w in updated_map.keys() if p[i][v[-1]][w] == c][0])
        V.remove(v[-1])

        n = len(v)
        possible_destinations = V.intersection(p[(i + 1) % n][v[-1]].keys())
        if (len(possible_destinations) == 0): 
            for j in range(n):
                if v[j] in p[n - 2][v[-2]] and v[0] in p[n - 1][v[j]] and v[-1] in p[j - 1][v[j - 1]] and v[j + 1] in p[j][v[-1]]:
                    total += p[n - 2][v[-2]][v[j]] + p[n - 1][v[j]][v[0]] + p[j - 1][v[j - 1]][v[-1]] + p[j][v[-1]][v[j + 1]]
                    total += - p[n - 2][v[-2]][v[-1]] - p[n - 1][v[-1]][v[0]] - p[j - 1][v[j - 1]][v[j]] - p[j][v[j]][v[j + 1]]
                    v[j], v[-1] = v[-1], v[j]
                    possible_destinations = V.intersection(p[(i + 1) % n][v[-1]].keys())
                    break

    total += p[N - 1][v[-1]][v[0]]
    return v, total

def kopt2(seq, total, p, V, starting_time, time_limit, T_0 = 0.01, beta = 0.992):
    T = T_0
    n = len(V)
    reset_to_best_counter = -1
    k = 0
    while True:
        if (abs(float(time() - starting_time) - float(time_limit)) <= 0.08): break

        if reset_to_best_counter == 0:
            total = best_total
            seq = best_seq
        elif reset_to_best_counter > 0:
            reset_to_best_counter -= 1

        i, j = tuple([randint(1, n - 1), randint(1, n - 1)])
        new_seq = list(seq)
        if not (new_seq[j] in p[i - 1][seq[i - 1]] and seq[(i + 1) % n] in p[i][new_seq[j]] and new_seq[i] in p[j - 1][seq[j - 1]] and seq[(j + 1) % n] in p[j][new_seq[i]]):
            continue;
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        new_total = 0
        for idx in range(n):
            new_total += p[idx][new_seq[idx]][new_seq[(idx + 1) % (n)]]

        sigma = new_total - total
        # print(total, new_total, sigma, prob)
        if sigma <= 0:
            # reset_to_best_counter = -1
            total = new_total
            seq = new_seq
        elif random() < exp(-sigma/T):
            k += 1
            best_seq = seq
            best_total = total
            reset_to_best_counter = 500

            total = new_total
            seq = new_seq
            # T = T*beta
            # T = T_0 * exp((1 - beta)*k)
            T = T_0 / k

    return seq, total

seq, total = getHC(p, set(V), s)
# for t in [0.001, 0.01, 0.1, 1, 10]:
#     for beta in [0.986, 0.988, 0.9885, 0.99, 0.992]:
#         new_seq, new_total = kopt2(seq, total, p, set(V), time(), time_limit, t, beta)
#         new_total -= len(V)*h
#         print(new_total, beta, t)

new_seq, new_total = kopt2(seq, total, p, set(V), starting_time, time_limit)
new_total -= len(V)*h
out = str(new_total) + "\r\n" + str(" ".join(new_seq))
output_file.write(out.strip())
output_file.close()
