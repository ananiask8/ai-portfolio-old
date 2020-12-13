#!/usr/bin/env python2

import gurobipy as g
import sys

# Create empty optimization model.
# In Python, only one environment exists and it is created internally 
# in the Model() constructor. 
model = g.Model()

# Parameters
# d = [int(val) for val in raw_input().split(" ")]
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")

d = [int(val) for val in input_file.readline().split(" ")]
T = len(d)
input_file.close()

# Create variables.
x = [model.addVar(lb=0, ub=g.GRB.INFINITY, vtype=g.GRB.INTEGER, name="x_" + str(i)) for i in range(T)]
z = [model.addVar(lb=0, ub=g.GRB.INFINITY, vtype=g.GRB.INTEGER, name="z_" + str(i)) for i in range(T)]

# Integrate new variables into model.
model.update()

# Set objective: maximize sum xi
model.setObjective(sum(z), sense=g.GRB.MINIMIZE)

# Add constraints.
for i in range(T):
	model.addConstr(z[i] >= sum([x[j] for j in range(i - 7, i + 1)]) - d[i], "cons_" + str(i) + "_pos")
	model.addConstr(z[i] >= d[i] - sum([x[j] for j in range(i - 7, i + 1)]), "cons_" + str(i) + "_neg")

model.write("out.lp")
# Solve the model.
model.optimize()

# Print the objective and the values of the decision variables in the solution.
# print(int(model.objVal))
# print(" ".join([str(int(x[i].x)) for i in range(T)]))
output_file.write(str(int(model.objVal)))
output_file.write("\r\n")
output_file.write(" ".join([str(int(x[i].x)) for i in range(T)]))
output_file.close()
