#!/usr/bin/env python2

import gurobipy as g
import sys
from collections import defaultdict

# Create empty optimization model.
# In Python, only one environment exists and it is created internally 
# in the Model() constructor. 
model = g.Model()

# Parameters
# d = [int(val) for val in raw_input().split(" ")]
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")

cs = input_file.readline().strip()
p = defaultdict(lambda : defaultdict(dict))
x = defaultdict(lambda : defaultdict(dict))
ci_total = defaultdict(lambda : defaultdict(dict))
cj_total = defaultdict(lambda : defaultdict(dict))
total = 0

for line in input_file.readlines():
    ci, cj, d, pijd = line.split(" ")
    ci_total[ci][d] = 0
    cj_total[cj][d] = 0
    p[d][ci][cj] = pijd
    x[d][ci][cj] = model.addVar(lb=0, ub=1, vtype=g.GRB.INTEGER, name=("x_%s_%s_%s" % (ci, cj, d)))
    total += pijd * x[d][ci][cj]
input_file.close()

# Integrate new variables into model.
model.update()

# Set objective: minimize total price spent
model.setObjective(total, sense=g.GRB.MINIMIZE)
for i in range(len(x)):
    d = str(i)
    v1 = x[d]
    per_day = 0
    try:
        for ci, v2 in v1.items():
            try:
                for cj, v3 in v2.items():
                    if (i == 0 and ci != cs) or (i == len(x) - 1 and cj != cs):
                        model.addConstr(x[d][ci][cj] == 0, "d_%s_%s_%s" % (ci, cj, d))
                    per_day += x[d][ci][cj]
                    ci_total[ci][d] += x[d][ci][cj]
                    cj_total[cj][d] += x[d][ci][cj]
                    # print("%s -> %s on day %s: $%s" % (ci, cj, d, p[d][ci][cj]))
            except AttributeError:
                pass
    except AttributeError:
        pass
    model.addConstr(per_day == 1, "cons_" + d)

for c in ci_total.keys():
    sum_ci = 0
    sum_cj = 0
    for i in range(len(x)):
        d = str(i)
        d_plus_1 = str(0 if i + 1 >= len(x) else i + 1)
        model.addConstr(cj_total[c][d] == ci_total[c][d_plus_1], "cons_" + c + "_" + d)
        sum_ci += ci_total[c][d]
        sum_cj += cj_total[c][d]
    model.addConstr(sum_ci == 1, "cons_ci_" + c)
    model.addConstr(sum_cj == 1, "cons_cj_" + c)

model.write("out.lp")
# Solve the model.
model.optimize()

# Print the objective and the values of the decision variables in the solution.
output_file.write(str(int(model.objVal)))
out = ""
for i in range(len(x)):
    d = str(i)
    v1 = x[d]
    try:
        for ci, v2 in v1.items():
            try:
                for cj, v3 in v2.items():
                    if x[d][ci][cj].x:
                        out += ("%s " % ci)
                        # print(x[d][ci][cj].x)
                        # print(p[d][ci][cj])
                        # print("%s -> %s on day %s: $%s" % (ci, cj, d, p[d][ci][cj]))
                        break
            except AttributeError:
                pass
    except AttributeError:
        pass

output_file.write("\r\n")
output_file.write(out.strip())
output_file.close()
