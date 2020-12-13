#!/usr/bin/env python2

import sys
from collections import defaultdict

# Parameters
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")
n = int(input_file.readline().strip())

# p -> processing time
# r -> release time
# d -> deadline
V = []
i = 0
for line in input_file.readlines():
    p, r, d = line.strip().split(" ")
    V.append((int(p), int(r), int(d), int(i)))
    i += 1
input_file.close()

def is_feasible(Vj, c):
    return all([c + p <= d for p, r, d, i in Vj])

def lower_bound(Vj, c):
    return max(c, min(Vj, key=lambda x: x[1])[1]) + sum([p for p, r, d, i in Vj])

def partial_solution_is_optimal(Vj, c):
    print("partial_solution_is_optimal", c <= min(Vj, key=lambda x: x[1])[1])
    return c <= min(Vj, key=lambda x: x[1])[1]

def bratley(V, c, visited, solution, best, root):
    current_best = None
    for v in V:
        if visited[v[3]] or c < root: continue

        solution.append(v)
        visited[v[3]] = True
        old_c, c = c, max(c, v[1]) + v[0]
        Vj = [u for u in V if not visited[u[3]]]
        # Python2 perserves latest assignment to var even if inside []
        # Had to rename from [v for v in V if not visited[v[3]]] to [u for u in V if not visited[u[3]]]

        if len(Vj) == 0:
            current_solution, current_cost, current_root = list(solution), c, root
            if len(current_solution) == len(V) and current_cost < best:
                best = current_cost
                current_best = (current_solution, current_cost, current_root)
                print([c[3] for c in current_solution], current_cost, root)
            break
        elif is_feasible(Vj, c) and lower_bound(Vj, c) < best:
            if partial_solution_is_optimal(Vj, c): root = c

            current_solution, current_cost, root = bratley(V, c, list(visited), list(solution), best, root)
            print([c[3] for c in current_solution], current_cost, root)
            if len(current_solution) == len(V) and current_cost < best:
                best = current_cost
                current_best = (current_solution, current_cost, root)        

        solution.pop()
        visited[v[3]] = False
        c = old_c

    if current_best is not None: return current_best
    else: return solution, c, root

def optimal_start_times(seq, n):
    solution = [0]*n
    c = 0
    for p, r, d, i in seq:
        solution[i] = max(c, r)
        c = solution[i] + p
    return solution

V = sorted(V, key=lambda x: x[2])
seq, last, root = bratley(V, 0, [False]*n, [], float("inf"), -1)
if len(seq) > 0:
    S = optimal_start_times(seq, n)
    out = ""
    for si in S: out += str(si) + "\r\n"
else: out = "-1"

output_file.write(out.strip())
output_file.close()
