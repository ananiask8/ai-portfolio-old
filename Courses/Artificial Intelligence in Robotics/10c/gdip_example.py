#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, math
import re
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch

sys.path.append('gdip')

#import dubins
from DubinsManeuver import DubinsManeuver as dubins

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

def plot_interval(point, interval, sensing_radius, turning_radius):
    ax = plt.gca()
    circle = Circle(point, sensing_radius, facecolor='yellow',edgecolor="orange", linewidth=1, alpha=0.2)
    ax.add_patch(circle)

    for angle in [interval[0], interval[0] + interval[1]]:
        end = point + turning_radius * np.array([math.cos(angle), math.sin(angle)])
        points = [point, end]
        plot_points(points, specs = 'y-')


def clear_plot():
    plt.clf()
    plt.axis('equal')

def save_figure(filename):
    plt.gcf().savefig(filename)

def rundom_select(data, number):
    idxs = np.array(range(len(data)))
    random.shuffle(idxs)
    return [data[i] for i in idxs[:number] ]

# Basic Dubins maneuver ######################################

turning_radius = 1

start = (0, 0, 1)
end = (5,0, 2)
step_size = 0.01 * turning_radius
dubins_path = dubins.shortest_path(start, end, turning_radius)
configurations, _ = dubins_path.sample_many(step_size)

clear_plot()
plot_points(configurations)

plt.pause(1)

# Dubins Interval Problem (DIP) ######################################

sensing_radius = 0.0

point1 = (0, 0)
point2 = (5,0)

interval1 = (1, 0.5)
interval2 = (2, 0.5)

for t in np.arange(0, 6, 0.1):
    interval2 = (2 + t, 0.5)
    step_size = 0.01 * turning_radius
    dubins_path = dubins.shortest_path_DIP(point1, interval1, point2, interval2, turning_radius)
    configurations, _ = dubins_path.sample_many(step_size)

    clear_plot()
    plot_interval(point1, interval1, sensing_radius, turning_radius)
    plot_interval(point2, interval2, sensing_radius, turning_radius)
    plot_points(configurations)

    plt.pause(0.1)

# Generalized Dubins Interval Problem (GDIP) ######################################

sensing_radius = 0.5

point1 = (0, 0)
point2 = (5,0)

interval1 = (1, 0.5)
interval2 = (2, 0.5)

for t in np.arange(0, 6, 0.1):
    interval2 = (2 + t, 0.5)
    step_size = 0.01 * turning_radius
    dubins_path = dubins.shortest_path_GDIP(point1, interval1, sensing_radius, point2, interval2, sensing_radius, turning_radius)
    configurations, _ = dubins_path.sample_many(step_size)

    clear_plot()
    plot_interval(point1, interval1, sensing_radius, turning_radius)
    plot_interval(point2, interval2, sensing_radius, turning_radius)
    plot_points(configurations)

    plt.pause(0.1)