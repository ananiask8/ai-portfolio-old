#!/usr/bin/env python2

import gurobipy as g
import numpy as np
import sys

# Parameters
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")
n, w, h = [int(val) for val in input_file.readline().strip().split(" ")]

# Init
model = g.Model()
model.Params.lazyConstraints = 1
s = [0]*n
for l in range(n):
    s[l] = [0]*h
    for j in range(h):
        s[l][j] = [0]*w
        for i in range(w):
            s[l][j][i] = np.array([0, 0, 0])

# Input
l = 0
for line in input_file.readlines():
    j, i, c = 0, 0, 0
    for val in line.strip().split(" "):
        if c != 0 and c % 3 == 0:
            i += 1
            if i != 0 and i % w == 0: j += 1
        s[l][j % h][i % w][c % 3] = val
        c += 1
    l += 1
input_file.close()

# Process input
dummy = n
d = [0]*(n + 1)
x = [0]*(n + 1)
for l in range(n + 1):
    d[l] = [0]*(n + 1) # +1 due to dummy node
    x[l] = [0]*(n + 1)

obj = 0
s = np.array(s)
for k1 in range(n + 1):
    for k2 in range(n + 1):
        x[k1][k2] = model.addVar(lb=0, ub=1, vtype=g.GRB.INTEGER, name=("x_%s_%s" % (k1, k2)))
        if k1 == k2: d[k1][k2] = g.GRB.INFINITY
        elif k1 < n and k2 < n: d[k1][k2] = np.sum(np.sum(abs(np.subtract(s[k1,:,w - 1], s[k2,:,0])), axis=1))
        else: d[k1][k2] = 0
        obj += d[k1][k2]*x[k1][k2]
x = np.array(x)

# Add constraints
model.update()
model.setObjective(obj, sense=g.GRB.MINIMIZE)
for k1 in range(n + 1):
    model.addConstr(sum(x[k1, :]) == 1, "d_i_%s" % k1)
    model.addConstr(sum(x[:, k1]) == 1, "d_%s_j" % k1)
    model.addConstr(x[k1, k1] == 0, "d_%s_%s" % (k1, k1))

def get_assignments(x, n):
    assignments = [(0, 0)]*n
    for i in range(n):
        for j in range(n):
            try: val = x[i][j].x
            except g.GurobiError: val = model.cbGetSolution(x[i][j])
            if round(val) == 1: assignments[i] = (j, x[i][j])

    return assignments

def get_tour(assignments, root):
    current = assignments[root]
    seq = []
    while not current[0] == root:
        seq.append(current)
        current = assignments[current[0]]
    seq.append(current)

    return seq

# Callback
def callback(x, n, root):
    def no_loops_lazy_constraint(model, where):
        if where == g.GRB.Callback.MIPSOL:
            assignments = get_assignments(x, n)
            for node in range(n):
                seq = get_tour(assignments, node)
                if len(seq) < n: model.cbLazy(sum([s[1] for s in seq]) <= len(seq) - 1)

    return no_loops_lazy_constraint


model.optimize(callback(x, n + 1, dummy))

# Print solution
assignments = get_assignments(x, n + 1)
seq = get_tour(assignments, dummy)

out = " ".join([str(s[0] + 1) for s in seq[:-1]]).strip()
# print(out)
output_file.write(out)
output_file.close()
