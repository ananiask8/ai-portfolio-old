# -*- coding: utf-8 -*-
"""

Dubins TSP with Neighborhoods (DTSPN)

@author: P.Vana
"""
import sys, os, time, math, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from DubinsManeuver import DubinsManeuver as dubins
import random

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

def dist_euclidean(coord1, coord2):
    (x1, y1) = coord1
    (x2, y2) = coord2
    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

def lowerPathGDIP(s1, s2, turning_radius):
    """
    Compute lower-bound path using GDIP between two samples

    Parameterseskatelevize.cz/
    ----------
    s1: start sample
    s2: end sample
    turning_radius: turning radius of the vehicle

    Returns
    -------
    Dubins maneuver
    """
    interval1 = [s1.alpha1, s1.alpha2 - s1.alpha1]
    interval2 = [s2.alpha1, s2.alpha2 - s2.alpha1]
    dubins_path = dubins.shortest_path_GDIP(s1.center, interval1, s1.radius, s2.center, interval2, s2.radius, turning_radius)
    return dubins_path

# cache results from computing length of the GDIP
already_computed = {}

def lowerPathGDIPLen(s1, s2, turning_radius):
    """
    Compute length of the lower-bound path using GDIP between two samples

    Parameters
    ----------
    s1: start sample
    s2: end sample
    turning_radius: turning radius of the vehicle

    Returns
    -------
    double - lenght of the path
    """
    key = (s1, s2)
    if key in already_computed:
        dist = already_computed[key]
    else:
        path = lowerPathGDIP(s1, s2, turning_radius)
        dist = path.get_length()
        # store results
        already_computed[key] = dist 
    
    return dist

def upperPathGDIP(s1, s2, turning_radius):
    """
    Compute feasible Dubins path two samples

    Parameters
    ----------
    s1: start sample
    s2: end sample
    turning_radius: turning radius of the vehicle

    Returns
    -------
    Dubins maneuver
    """
    q1 = s1.getFeasibleState()
    q2 = s2.getFeasibleState()
    dubins_path = dubins.shortest_path(q1, q2, turning_radius)
    return dubins_path

def upperPathGDIPLen(s1, s2, turning_radius):
    """
    Compute feasible Dubins path two samples

    Parameters
    ----------
    s1: start sample
    s2: end sample
    turning_radius: turning radius of the vehicle

    Returns
    -------
    double - lenght of the path
    """
    path = upperPathGDIP(s1, s2, turning_radius)
    return path.get_length()

def retrieve_path(samples, dst_fce, turning_radius, selected_samples):
    n = len(samples)
    path = []
    for a in range(0,n):
        b = (a+1) % n
        g1 = samples[a][selected_samples[a]]
        g2 = samples[b][selected_samples[b]]
        path.append(dst_fce(g1, g2, turning_radius))
    return path

def path_len(path):
    length = 0
    for dub in path:
        length += dub.get_length()
    return length

def plot_path(path, turning_radius, settings):
    step_size = 0.01 * turning_radius
    for dub in path:
        configurations, _ = dub.sample_many(step_size)
        plot_points(configurations, settings) 

##################################################
# Target region given by the location and its sensing radius
##################################################
class TargetRegion:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_position_at_boundary(self, beta):
        return self.center + self.radius * np.array([math.cos(beta), math.sin(beta)])

##################################################
# Sample on the given target region
##################################################
class Sample:
    def __init__(self, targetRegion):
        # reference to the specific target region of the sample
        self.target = targetRegion

        # heading angle interval
        self.alpha1 = 0
        self.alpha2 = 2 * math.pi
        self.alphaResolution = 1
        # position (on the boundary) interval
        self.beta1 = 0
        self.beta2 = 2 * math.pi
        self.betaResolution = 1

        # center and radius of the position neighborhood on the boundary of the target region
        self.center = np.array(targetRegion.center)
        self.radius = targetRegion.radius

    def split(self, resolution):
        """
        Split the actual sample into two new ones.
        The first is stored directly in the actual sample, and the second is returned.
        If the required resolution is already met, then nothing is done and None is returned.

        Parameters
        ----------
        resolution: the requred resolution

        Returns
        -------
        Sample - the second new sample
        """
        # prefer splitting according position resolution
        if self.betaResolution < resolution:
            sam1 = copy.copy(self)
            sam2 = copy.copy(self)
            sam1.betaResolution = sam2.betaResolution = 2 * self.betaResolution
            sam1.beta2 = sam2.beta1 = (self.beta1 + self.beta2) / 2
            sam1.update_center_radius()
            sam2.update_center_radius()
            return [sam1, sam2]
        if self.alphaResolution < resolution:
            sam1 = copy.copy(self)
            sam2 = copy.copy(self)
            sam1.alphaResolution = sam2.alphaResolution = 2 * self.alphaResolution
            sam1.alpha2 = sam2.alpha1 = (self.alpha1 + self.alpha2) / 2
            return [sam1, sam2]
        return None

    def update_center_radius(self):
        p1 = self.target.get_position_at_boundary(self.beta1)
        p2 = self.target.get_position_at_boundary(self.beta2)
        self.center = (p1 + p2) / 2
        self.radius = dist_euclidean(p1, p2) / 2

    def getFeasibleState(self):
        pos = self.target.get_position_at_boundary(self.beta1)
        q = np.zeros(3)
        q[0:2] = pos
        q[2] = self.alpha1
        return q

    def plot(self):
        ax = plt.gca()
        circle = Circle(self.center, self.radius, facecolor=None ,edgecolor="green", linewidth=1, alpha=0.2)
        ax.add_patch(circle)

##################################################
# Sampling structure which holds all the used samples
##################################################
class Sampling:
    def __init__(self, centers, sensingRadius):
        self.targets = [TargetRegion(c, sensingRadius) for c in centers]
        self.samples = [[Sample(t)] for t in self.targets]

    def refine_samples(self, selected, resolution):
        """
        Refine the seleted samples if the required resolution is not met.

        Parameters
        ----------
        slected: indexes of the selected samples (vector 1 x n)
        resolution: the requred resolution

        Returns
        -------
        boolean - true if any sample is refined
        """
        n = len(self.samples)
        refined = False
        for i in range(n):
            to_split = selected[i]
            samp = self.samples[i][to_split]
            res = samp.split(resolution)
            if not res is None:
                self.samples[i][to_split] = res[0]
                self.samples[i].append(res[1])
                refined = True 
        return refined

##################################################
# The main solver class
##################################################
class GDIPSolver:
    def __init__(self, turning_radius, goals, sensing_radius):
        self.turning_radius = turning_radius
        self.sensing_radius = sensing_radius
        self.goals = goals
        self.sampling = Sampling(goals, sensing_radius)

        self.lower_path = []
        self.upper_path = []

        self.lower_bound = 0
        self.upper_bound = float('inf')
    
    def plot_map(self):
        plt.clf()
        plt.axis('equal')
        plot_points(self.goals, 'ro')
        if self.sensing_radius != None:
            for goal in self.goals:
                plot_circle(goal, self.sensing_radius)

    def plot_tour_and_return_length(self, selected_samples, maneuver_function, color):
        sampling = self.sampling
        n = len(self.sampling.samples)
        step_size = 0.01 * self.turning_radius
        length = 0
        for a in range(0,n):
            b = (a+1) % n
            g1 = sampling.samples[a][selected_samples[a]]
            g2 = sampling.samples[b][selected_samples[b]]

            path = maneuver_function(g1, g2, self.turning_radius)
            length += path.get_length()
            configurations, _ = path.sample_many(step_size)
            plot_points(configurations, color)
        return length

    def plot_actual_and_return_bounds(self):
        """
        Plot the actual sampling, lower and upper bound path

        Returns
        -------
        (double, double) - lower bound, upper bound
        """
        self.plot_map()

        for s in self.sampling.samples:
            for ss in s:
                ss.plot()

        lower_selected_samples = self.find_lower_bound_tour()
        upper_selected_samples = self.find_upper_bound_tour()

        lower_bound = self.plot_tour_and_return_length(lower_selected_samples, lowerPathGDIP, 'r-')
        upper_bound = self.plot_tour_and_return_length(upper_selected_samples, upperPathGDIP, 'b-')
        return (lower_bound, upper_bound)
    
    def find_lower_bound_tour(self, limit=100):
        """
        Select the samples which represent the shortest lower bound tour

        Returns
        -------
        indexes of the samples (vector 1 x n)
        """
        sampling, turning_radius = self.sampling, self.turning_radius
        n = len(self.sampling.samples)
        # init
        if len(self.upper_path) == 0:
            selected_samples = [0]*n
            best = sum([lowerPathGDIPLen(sampling.samples[i][0], sampling.samples[(i+1)%n][0], turning_radius) for i in range(n)])
        else:
            selected_samples = self.upper_path
            best = sum([lowerPathGDIPLen(sampling.samples[i][selected_samples[i]], sampling.samples[(i+1)%n][selected_samples[(i+1)%n]], turning_radius) for i in range(n)])

        improving = True
        count = 0
        indeces = list(range(n))
        np.random.shuffle(indeces)
        while improving:
            # if count > limit: break
            count += 1
            improving = False
            for j in indeces:
                i = (j-1) % n
                k = (j+1) % n
                prev_idx, current_idx, next_idx = i, j, k
                prev_r, current_r, next_r = sampling.samples[prev_idx], sampling.samples[current_idx], sampling.samples[next_idx]
                prev_v_idx, current_v_idx, next_v_idx = selected_samples[i], selected_samples[j], selected_samples[k]
                prev_v, current_v, next_v = prev_r[prev_v_idx], current_r[current_v_idx], next_r[next_v_idx]
                current_add_component = lowerPathGDIPLen(prev_v, current_v, turning_radius) + lowerPathGDIPLen(current_v, next_v, turning_radius)

                m = len(current_r)
                for new_v_idx in range(m):
                    new_v = current_r[new_v_idx]
                    new_add_component = lowerPathGDIPLen(prev_v, new_v, turning_radius) + lowerPathGDIPLen(new_v, next_v, turning_radius)
                    if best - current_add_component + new_add_component < best and random.random() > 0.5:
                        improving = True
                        best += new_add_component - current_add_component
                        selected_samples[j] = new_v_idx
                        break # necessary, or save new add component to current add component
        self.lower_path = selected_samples
        return selected_samples

    def find_upper_bound_tour(self, limit=100):
        """
        Select the samples which represent the shortest upper bound (feasible) tour

        Returns
        -------
        indexes of the samples (vector 1 x n)
        """
        # TODO - insert your code here
        # take lower bound path and refine indeces selected
        # find now a feasible solution in this graph
        sampling, turning_radius = self.sampling, self.turning_radius
        n = len(self.sampling.samples)
        # init
        if len(self.upper_path) == 0:
            selected_samples = [0]*n
            best = sum([upperPathGDIPLen(sampling.samples[i][0], sampling.samples[(i+1)%n][0], turning_radius) for i in range(n)])
        else:
            selected_samples = self.upper_path
            best = sum([upperPathGDIPLen(sampling.samples[i][selected_samples[i]], sampling.samples[(i+1)%n][selected_samples[(i+1)%n]], turning_radius) for i in range(n)])

        improving = True
        count = 0
        indeces = list(range(n))
        np.random.shuffle(indeces)
        while improving:
            # if count > limit: break
            count += 1
            improving = False
            for j in indeces:
                i = (j-1) % n
                k = (j+1) % n
                prev_idx, current_idx, next_idx = i, j, k
                prev_r, current_r, next_r = sampling.samples[prev_idx], sampling.samples[current_idx], sampling.samples[next_idx]
                prev_v_idx, current_v_idx, next_v_idx = selected_samples[i], selected_samples[j], selected_samples[k]
                prev_v, current_v, next_v = prev_r[prev_v_idx], current_r[current_v_idx], next_r[next_v_idx]
                current_add_component = upperPathGDIPLen(prev_v, current_v, turning_radius) + upperPathGDIPLen(current_v, next_v, turning_radius)

                m = len(current_r)
                for new_v_idx in range(m):
                    new_v = current_r[new_v_idx]
                    new_add_component = upperPathGDIPLen(prev_v, new_v, turning_radius) + upperPathGDIPLen(new_v, next_v, turning_radius)
                    if best - current_add_component + new_add_component < best and random.random() > 0.5:
                        improving = True
                        best += new_add_component - current_add_component
                        selected_samples[j] = new_v_idx
                        break # necessary, or save new add component to current add component
        self.upper_path = selected_samples
        return selected_samples

