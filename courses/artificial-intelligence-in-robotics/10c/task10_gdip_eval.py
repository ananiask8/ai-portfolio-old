#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import re
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch

sys.path.append('dtspn')
sys.path.append('lkh')
sys.path.append('gdip')

import GDIPSolver as gdipsolver

##################################################
# Helper functions
##################################################
def pause(time = 1):
	plt.pause(time)

def save_figure(filename):
	plt.gcf().savefig(filename)

################################################
# Testing
################################################
#define planning problems:
    #  map file 
    #  minimum turning radius
    #  sensing radius
	#  solver type
scenarios = [
	("./problems/gdip-n10.txt", 1, 1, 'DTSPN-GDIP'),
]

for scenario in scenarios:
	# Fix the problems
	random.seed(42)

	# read config with goal positions
	goals = []
	with open(scenario[0]) as fp:
		for line in fp:
			label, x, y = line.split()
			goals.append((float(x), float(y)))

	turning_radius = scenario[1]
	sensing_radius = scenario[2]
	solver_type = scenario[3]

	######################################
	# plot arena and goals (and their neighborhoods)
	######################################
	plt.pause(0.1)

	######################################
	#tour planning
	######################################

	solver = gdipsolver.GDIPSolver(turning_radius, goals, sensing_radius)

	solver.plot_actual_and_return_bounds()

	os.system("mkdir -p images")

	max_resolution = 64
	act_res = 4
	while act_res <= max_resolution:
		refined = True
		while refined:
			selected_samples = solver.find_lower_bound_tour()
			refined = solver.sampling.refine_samples(selected_samples, act_res)
		(lower_bound, upper_bound) = solver.plot_actual_and_return_bounds()
		plt.title("Maximum resolution: {:4d}".format(act_res))
		gap = (upper_bound - lower_bound) / upper_bound * 100.0
		print("Resolution: {:4d} Lower bound: {:6.2f} Upper bound (feasible): {:6.2f} Gap(%): {:6.2f}".format(act_res, lower_bound, upper_bound, gap))
		save_figure("images/resolution-{}.png".format(act_res))
		pause(0.001)
		act_res *= 2

	pause(2)

