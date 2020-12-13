#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import re
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch

sys.path.append('dtspn')
sys.path.append('lkh')
sys.path.append('gdip')

import DTSPNSolver as dtspn

##################################################
# Helper functions
##################################################
def pause(time = 1):
	plt.pause(time)

def plot_points(points, specs = 'b'):
	x_val = [x[0] for x in points]
	y_val = [x[1] for x in points]
	plt.plot(x_val, y_val, specs)

def plot_circle(xy, radius):
	ax = plt.gca()
	circle = Circle(xy, radius, facecolor='yellow',edgecolor="orange", linewidth=1, alpha=0.2)
	ax.add_patch(circle)

def plot_map():
	plt.clf()
	plt.axis('equal')
	plot_points(goals, 'ro')
	if sensing_radius != None:
		for goal in goals:
			plot_circle(goal, sensing_radius)

def save_figure(filename):
	plt.gcf().savefig(filename)

################################################
# Testing
################################################
if __name__ == '__main__':
	solver_type = 'DTSPN-AA'
	# python3 planner-dtsp.py ./problems/burma14.txt --sensing-distance 0.5 --dubins-radius 0.5 output.txt
	# python3 planner-dtsp.py ./problems/att48.txt --sensing-distance 200 --dubins-radius 300 output.txt
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help='Provide the path to the file containing the instance to solve.')
	parser.add_argument('output', help='Provide a path to write the output into.')

	parser.add_argument('--dubins-radius', help='Provide the turning radius of the Dubins vehicles.', type=float, default=((8.33**2)/2.0))
	parser.add_argument('--sensing-distance', help='Provide the sensing radius around the goal locations.', type=float, default=0)
	parser.add_argument('--heading-samples', help='Provide headings to sample per location in neighborhood.', type=int, default=8)
	parser.add_argument('--neighborhood-samples', help='Provide the amount of samples to take from each sensing region.', type=int, default=8)
	args = parser.parse_args()

	# Fix the problems
	random.seed(42)

	# read config with goal positions
	goals = []
	with open(args.input) as fp:
		for line in fp:
			arr = line.split()
			if len(arr) == 2:
				x, y = arr
			else:
				_, x, y = line.split()

			goals.append((float(x), float(y)))

	turning_radius = args.dubins_radius
	sensing_radius = args.sensing_distance
	neighborhood_samples = args.neighborhood_samples
	heading_samples = args.heading_samples

	######################################
	# plot arena and goals (and their neighborhoods)
	######################################
	plot_map()
	plt.pause(0.1)

	######################################
	#tour planning
	######################################

	solver = dtspn.DTSPNSolver(step_size=0.2)
	if solver_type == 'DTSPN-decoupled':
		path = solver.plan_tour_decoupled(goals, sensing_radius, turning_radius, neighborhood_samples, heading_samples)
	if solver_type == 'DTSPN-AA':
		path = solver.plan_tour_aa(goals, sensing_radius, turning_radius, neighborhood_samples, heading_samples)

	######################################
	# write to file
	######################################
	out = ''
	for step in path:
		out += "{} {} {} {}\r\n".format(step[0], step[1], 5.0, step[2])
	f = open(args.output, 'w+')
	f.write(out)

	######################################
	# plot result
	######################################
	plot_points(path, 'b-')
	pause(2)
	input()

	######################################
	# execute in simulator
	######################################
	start = 'roslaunch simulation simulation.launch gui:=true'
	spawn = 'waitForSimulation; spawn 1 --run --delete --enable-rangefinder --enable-ground-truth'
	trajectory = 'export UAV_NAME=uav1; roslaunch trajectory_handler single_trajectory_from_file.launch current_working_directory:=`pwd`/ file:={}'.format(args.output)
	launch1 = 'rosservice call /uav1/mavros/cmd/arming 1'
	launch2 = 'rosservice call /uav1/control_manager/motors 1'
	launch3 = 'rosservice call /uav1/mavros/set_mode 0 offboard'
	launch4 = 'rosservice call /uav1/mav_manager/takeoff'
	os.system(start)
	os.system(spawn)
	os.system(trajectory)
	os.system(launch1)
	os.system('sleep 1')
	os.system(launch2)
	os.system(launch3)
	os.system('sleep 1')
	os.system(launch4)
