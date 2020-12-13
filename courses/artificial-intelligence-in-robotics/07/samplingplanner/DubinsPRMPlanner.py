#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import itertools

import collections
import heapq
import matplotlib.pyplot as plt
import dubins as dubins

import Environment as env
from Graph import Graph
from pprint import pprint as pp

class DubinsPRMPlanner:
    def __init__(self, limits, max_step_fraction_of_largest_dimension = 1/256.0, max_axis_orientation_change = np.pi/7.0):
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
        self.rho = np.linalg.norm([x_upper-x_lower, y_upper-y_lower, z_upper-z_lower])/len(linear_dof)

    def nearest_neighbor_idx(self, dataset, v, k=1, **kwargs):
        result, _ = self.flann.nn(np.array(dataset, dtype='float64'), np.array(v, dtype='float64'), k, **kwargs)
        return result[0]

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

        # Running for only one heading (the sampled one) because finding a feasible one here doesn't
        # help me to connect sampled nodes: I need to test for same heading.
        # 
        # X WRONG X headings = np.linspace(q1[2], q1[2] + 2.*math.pi, 50) X WRONG X
        headings = np.linspace(q1[2], q1[2] + 2.*math.pi, 50) if sample else [q1[2]]
        np.random.shuffle(headings)

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

    def build_graph(self, samples, new_samples, env):
        g = Graph()
        g = self.add_edges_to_graph(samples, new_samples, env, g)

        return g

    def add_edges_to_graph(self, samples, new_samples, env, g):
        adj_list = []
        new = []
        for i in range(len(samples)):
            sample_i = samples[i]
            for j in range(len(samples)):
                sample_j = samples[j]
                distance = np.linalg.norm(sample_i[0:3] - sample_j[0:3])
                if distance <= self.rho:
                    free, _ = self.path_check(sample_i, sample_j, env)
                    if free:
                        adj_list.append((tuple(sample_i), tuple(sample_j), distance))
                        # 
                        # 
                        # I can add only in one direction when having non-holonomic vehicles:
                        # it is not the same to have (x1,y1,phi1) -> (x2,y2,phi2)
                        # Ex: Imagine two points in the same vertical line, facing opposite directions
                        # [ ][ ][ ][  ][  ][ o1][--][->][ ][ ]
                        # [ ][ ][ ][  ][  ][   ][  ][  ][ ][ ]
                        # [ ][ ][X][<-][--][o2 ][  ][  ][ ][ ]
                        # where the "o-->" represent a vector sample, and the X represents an obstacle
                        # Clearly, there exist some radius for which going from o1 to o2 is possible
                        # But it is not possible to go from o2 to o1, for that same radius
                        # ■
                        # 
                        # X WRONG X adj_list.append((tuple(sample_j), tuple(sample_i), distance)) X WRONG X

        g.add_edges(adj_list)
        return g

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

        samples = self.sample_config_space(n=50)
        samples = np.vstack((samples, [start]))
        samples = np.vstack((samples, [goal]))
        samples = list(filter(lambda sample: not environment.check_robot_collision(self.to_pose_matrix(sample)), samples))
        g = self.build_graph(samples, samples, environment)

        path = None
        i = 0
        while path is None:
            if len(samples) == 1000: return []
            print("Iteration #{}: {} samples, |V(G)| = {}, |E(G)| = {}".format(i, len(samples), len(g.vertices), len(g.edges)))

            i += 1
            try:
                path = g.dijkstra(tuple(start), tuple(goal))
                # print(path, tuple(goal), path[0] == tuple(goal))
                if path[0] == tuple(goal):
                    path = None
                    raise
            except:
                # samples_cfree = list(filter(lambda sample: not environment.check_robot_collision(self.to_pose_matrix(sample)), self.sample_config_space(n=30)))
                # _ = self.add_edges_to_graph(samples, samples_cfree, environment, g)
                # samples = np.vstack((samples, samples_cfree))
                
                samples = self.sample_config_space(n=i*50)
                samples = np.vstack((samples, [start]))
                samples = np.vstack((samples, [goal]))
                samples = list(filter(lambda sample: not environment.check_robot_collision(self.to_pose_matrix(sample)), samples))
                g = self.build_graph(samples, samples, environment)

        plan = [start]
        print(path)
        for i,j in zip(range(len(path) - 1), range(1, len(path))):
            _, sequence = self.path_check(np.array(path[i]), np.array(path[j]), environment)
            plan += sequence
        return([self.to_pose_matrix(node) for node in plan])

    def rotation_matrix(self, yaw, pitch, roll):
        """
        Constructs rotation matrix given the euler angles
        yaw = rotation around z axis
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

