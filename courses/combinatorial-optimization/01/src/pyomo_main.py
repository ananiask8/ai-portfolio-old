
import gurobipy as gp
import math

m = gp.Model()
x = m.addVar(name="x", ub=1)
y = m.addVar(name="y", lb=-gp.GRB.INFINITY)

# Create x-points and y-points for approximation of y = x*log(x)
xs = [0.01*i for i in range(101)]
ys = [p*math.log(p) if p != 0 else 0 for p in xs]

# Add piecewise-linear constraint
m.addGenConstrPWL(x, y, xs, ys, "pwl")

# Minimize approximation of y = x*log(x)
m.setObjective(y, gp.GRB.MINIMIZE)
m.optimize()

# from __future__ import division
# import pyomo.environ as pyo
# import sys

# # IO
# input_file = open(sys.argv[1], "r")
# output_file = open(sys.argv[2], "w")
# vector = input_file.readline().split(" ")
# input_file.close()
# T = len(vector)

# # Init
# model = pyo.ConcreteModel()
# model.T = pyo.RangeSet(0, T - 1, ordered=True)
# model.d = pyo.Param(model.T, initialize={i: int(vector[i]) for i in range(len(vector))}, domain=pyo.Reals)

# # Variables
# model.x = pyo.Var(model.T, domain=pyo.NonNegativeIntegers)
# model.z = pyo.Var(model.T, domain=pyo.NonNegativeIntegers)

# # Objective
# model.OBJ = pyo.Objective(expr=sum(pyo.log(model.z[i]) for i in model.T))
# # model.OBJ = pyo.Objective(expr=pyo.summation(model.z))

# # Constraints
# def constraint_rule_1(model, i):
# 	return model.z[i] >= sum([model.x[j % T] for j in range(i - 7, i + 1)]) - model.d[i]

# def constraint_rule_2(model, i):
# 	return model.z[i] >= model.d[i] - sum([model.x[j % T] for j in range(i - 7, i + 1)])

# model.Constraint1 = pyo.Constraint(model.T, rule=constraint_rule_1)
# model.Constraint2 = pyo.Constraint(model.T, rule=constraint_rule_2)

# # Solve
# opt = pyo.SolverFactory('gurobi')
# opt.solve(model)

# # IO
# output_file.write(str(int(pyo.value(model.OBJ))))
# output_file.write("\r\n")
# output_file.write(" ".join([str(int(pyo.value(model.x[i]))) for i in range(T)]))
# output_file.close()