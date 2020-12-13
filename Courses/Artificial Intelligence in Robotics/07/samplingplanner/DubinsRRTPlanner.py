#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
from pyflann import *
import itertools
import dubins as dubins

import collections
import heapq
import matplotlib.pyplot as plt

from Graph import Graph
import Environment as env

class DubinsRRTPlanner:

    def __init__(self, limits, max_step_fraction_of_largest_dimension = 1/256.0, max_axis_orientation_change = np.pi/8.0):
        """
        Parameters
        ----------
        limits: list((float, float))
            translation limits in individual axes 
        max_step_fraction_of_largest_dimension: float
            fraction of the largest configuration dimension which limits the max translation in axes x,y,z
        max_axis_orientation_change: float
            max change in orientation between two consecutive path points
        """
        x_lower = limits[0][0] 
        x_upper = limits[0][1]
        y_lower = limits[1][0]
        y_upper = limits[1][1]
        z_lower = limits[2][0]
        z_upper = limits[2][1]
        self.radius = 0.5
        self.limits = limits
        _, linear_dof = zip(*filter(lambda x: not np.abs(x[0][0] - x[0][1]) == 0, zip(limits[:3], range(len(limits[:3])))))
        self.max_translation = 3*[max_step_fraction_of_largest_dimension * np.max([x_upper-x_lower, y_upper-y_lower, z_upper-z_lower])]
        self.max_translation += 3*[max_axis_orientation_change]
        self.rho = np.linalg.norm([x_upper-x_lower, y_upper-y_lower, z_upper-z_lower])/(2*len(linear_dof))
        self.flann = FLANN()

    def nearest_neighbor_idx(self, dataset, v, k=1, **kwargs):
        # can improve distance measure by using what dubins library returns?
        result, dist = self.flann.nn(np.array(dataset, dtype='float64'), np.array(v, dtype='float64'), k, **kwargs)
        return result[0], dist[0]

    def to_pose_matrix(self, v):
        R = self.rotation_matrix(v[5],v[4],v[3])
        T = v[0:3]
        P = np.hstack((R,T.reshape((3,1))))
        P = np.vstack((P,[0,0,0,1]))

        return P

    def sample_config_space(self, n = 30):
        dof = len(self.limits)
        samples = np.random.rand(dof,n)
        i = 0
        for limit in self.limits: #for each DOF in configuration 
            scale = limit[1] - limit[0] #calculate the scale
            samples[i,:] = samples[i,:]*scale + limit[0] #scale and shift the random samples
            i += 1

        # NOTE
        # samples are being generated inside the limits.
        # the translations close from point 1 to point 2, they never go above the values of point 2, hence, the limits are held.
        return np.transpose(samples)

    def to6dof(self, configurations):
        new_configurations = []
        for configuration in configurations:
            new = np.hstack((configuration[0:2], np.array([0]*3), configuration[-1]))
            new_configurations.append(new)
        return new_configurations

    def in_bounds(self, configurations):
        for configuration in configurations:
            for axis in range(len(self.limits)):
                if not(self.limits[axis][0] <= configuration[axis] and configuration[axis] <= self.limits[axis][1]): return False

        return True

    def any_collision(self, configurations, env):
        for node in configurations:
            if env.check_robot_collision(self.to_pose_matrix(node)): return True

        return False

    def path_check(self, start, goal, env, sample = False):
        q0 = np.hstack((start[0:2], start[:-1]))
        q1 = np.hstack((goal[0:2],goal[:-1]))

        headings = np.linspace(q1[2], q1[2] + 2.*math.pi, 50) if sample else [q1[2]]
        # np.random.shuffle(headings)
        for heading in headings: # for [0,2π] heading range check, when using a step of π/50
            q1[2] = heading

            # calcuating the shortest path between two configurations
            path = dubins.shortest_path(q0, q1, self.radius)

            # sampling the path
            configurations, _ = path.sample_many(self.max_translation[0])
            configurations = self.to6dof(configurations)
            if self.in_bounds(configurations) and not self.any_collision(configurations, env):
                return True, configurations

        return False, []

    def plan(self, environment, start, goal):
        """
        Method to plan the path

        Parameters
        ----------
        environment: Environment
            Map of the environment that provides collision checking 
        start: numpy array (6x1)
            start configuration of the robot (x,y,z,phi_x,phi_y,phi_z)
        goal: numpy array (6x1)
            goal configuration of the robot (x,y,z,phi_x,phi_y,phi_z)

        Returns
        -------
        list(numpy array (4x4))
            the path between the start and the goal Pose in SE(3) coordinates
        """
        start = np.array(start, dtype='float64')
        goal = np.array(goal, dtype='float64')

        g = Graph()
        samples = np.array([start])
        nn, sample = None, None
        i = 0
        while sample is None or not self.path_check(sample, goal, environment)[0]:
            if len(samples) == 5000: return []
            print("Iteration #{}: {} samples, |V(G)| = {}, |E(G)| = {}".format(i, len(samples), len(g.vertices), len(g.edges)))

            i += 1
            sample = self.sample_config_space(n=1)[0]
            idx, distance = self.nearest_neighbor_idx(samples, sample)
            nn = samples[idx]
            while environment.check_robot_collision(self.to_pose_matrix(sample)) or \
                    not self.path_check(nn, sample, environment)[0]:
                sample = self.sample_config_space(n=1)[0]
                idx, distance = self.nearest_neighbor_idx(samples, sample)
                nn = samples[idx]

            samples = np.vstack((samples, sample))
            g.add_edges([
                (tuple(sample), tuple(nn), distance),
                (tuple(nn), tuple(sample), distance)
            ])

        g.add_edges([
            (tuple(sample), tuple(goal), distance),
            (tuple(goal), tuple(sample), distance)
        ])
        path = g.dijkstra(tuple(start), tuple(goal))
        plan = []
        # print("path length " + str(len(path)))
        print(path)
        for i,j in zip(range(len(path) - 1), range(1, len(path))):
            # print(path[i], path[j])
            _, sequence = self.path_check(np.array(path[i]), np.array(path[j]), environment)
            # print(sequence)
            plan += sequence
            # break
        # print(plan)
        # print(len(plan))
        # print(plan[-1])
        # print(len(list(filter(lambda x: x[0] < self.limits[0][0] or x[0] > self.limits[0][1] or x[1] < self.limits[1][0] or x[1] > self.limits[1][1] or x[2] < self.limits[2][0] or x[2] > self.limits[2][1], plan))))
        return([self.to_pose_matrix(node) for node in plan])
        # return []
    
    def rotation_matrix(self, yaw, pitch, roll):
        """
        Constructs rotation matrix given the euler angles
        yaw = rotation afloor z axis
        pitch = rotation around y axis
        roll = rotation around x axis

        Parameters
        ----------
        yaw: float
        pitch: float
        roll: float
            respective euler angles
        """
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(roll), math.sin(roll)],
                        [0, -math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, -math.sin(pitch)],
                        [0, 1, 0],
                        [math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), math.sin(yaw), 0],
                        [-math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])

        R = R_x.dot(R_y).dot(R_z)
        return R

if __name__ == '__main__':
    pyflann()