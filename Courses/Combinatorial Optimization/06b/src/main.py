#!/usr/bin/env python2

import sys
from collections import defaultdict
from random import randint
import copy

# Parameters
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")

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
    n = len(V)
    v = [s]
    total = 0
    V.remove(s)
    for i in range(n - 1):
        possible_destinations = V.intersection(set(p[i][v[-1]].keys()))
        updated_map = {k: v for k, v in p[i][v[-1]].items() if k in possible_destinations}
        c = min(updated_map.values())
        total += c
        v.append([w for w in updated_map.keys() if p[i][v[-1]][w] == c][0])
        V.remove(v[-1])

    total += p[n - 1][v[-1]][v[0]]
    return v, total

def kopt2(seq, total, p, V, limit = 500):
    n = len(V)
    for iteration in range(limit):
        i, j = tuple([randint(1, n - 1), randint(1, n - 1)])
        new_seq = list(seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_total = 0
        for idx in range(n):
            new_total += p[idx][new_seq[idx]][new_seq[(idx + 1) % (n)]]


        if total > new_total:
            total = new_total
            seq = new_seq

    return seq, total

seq, total = getHC(p, set(V), s)
seq, total = kopt2(seq, total, p, set(V))
total -= len(V)*h

# Print the objective and the values of the decision variables in the solution.
out = str(total) + "\r\n" + str(" ".join(seq))
output_file.write(out.strip())
output_file.close()
