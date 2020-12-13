#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import itertools

import collections
import heapq
import matplotlib.pyplot as plt

import Environment as env
from Graph import Graph
from pprint import pprint as pp

class PRMPlanner:

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
        self.limits = limits
        _, linear_dof = zip(*filter(lambda x: not np.abs(x[0][0] - x[0][1]) == 0, zip(limits[:3], range(len(limits[:3])))))
        self.max_translation = 3*[max_step_fraction_of_largest_dimension * np.max([x_upper-x_lower, y_upper-y_lower, z_upper-z_lower])]
        self.max_translation += 3*[max_axis_orientation_change]
        self.rho = np.linalg.norm([x_upper-x_lower, y_upper-y_lower, z_upper-z_lower])/len(linear_dof)

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

    def generate_translation_samples(self, start, goal, axis=0):
        samples = [start]
        for val in np.linspace(start[axis], goal[axis], np.round(np.abs(goal[axis] - start[axis])/self.max_translation[axis])):
            pos = np.copy(samples[-1])
            pos[axis] = val
            samples.append(pos)

        pos = np.copy(samples[-1])
        pos[axis] = goal[axis]
        samples.append(pos)
        return samples[1:]

    def path_check(self, start, goal, env):
        samples = [start]
        collision = False
        while np.any(np.abs(goal - samples[-1]) >= self.max_translation):
            new_sample = np.array([samples[-1][axis] + np.sign(goal[axis] - samples[-1][axis])*min(abs(goal[axis] - samples[-1][axis]), self.max_translation[axis]) for axis in range(len(samples[-1]))])
            if env.check_robot_collision(self.to_pose_matrix(new_sample)):
                collision = True
                break
            samples.append(new_sample)
        if not collision: return True, samples

        _, dof = zip(*filter(lambda x: not np.abs(x[0][0] - x[0][1]) == 0, zip(self.limits, range(len(self.limits)))))
        for order in itertools.permutations(dof):
            samples = [start]
            collision = False
            for axis in order:
                new_samples = self.generate_translation_samples(samples[-1], goal, axis)
                if np.any([env.check_robot_collision(self.to_pose_matrix(sample)) for sample in new_samples]):
                    collision = True
                    break
                samples += new_samples
            if not collision: break
        return not collision, samples

    def build_graph(self, samples, new_samples, env):
        g = Graph()
        g = self.add_edges_to_graph(samples, new_samples, env, g)

        return g

    def add_edges_to_graph(self, samples, new_samples, env, g):
        adj_list = []
        for sample_i in samples:
            for sample_j in new_samples:
                distance = np.linalg.norm(np.abs(sample_i[0:3] - sample_j[0:3]))
                if np.all(sample_i == sample_j) or distance > self.rho: break 
                
                free, _ = self.path_check(sample_i, sample_j, env)
                if not free: break
                adj_list.append((tuple(sample_i), tuple(sample_j), distance))
                adj_list.append((tuple(sample_j), tuple(sample_i), distance))
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

        samples = self.sample_config_space(n=60)
        samples = np.vstack((samples, [start]))
        samples = np.vstack((samples, [goal]))
        samples = list(filter(lambda sample: not environment.check_robot_collision(self.to_pose_matrix(sample)), samples))
        g = self.build_graph(samples, samples, environment)
            
        path = None
        i = 0
        while path is None or len(path) <= 1:
            print(i)
            if i == 100: return []
            i += 1
            # print(len(samples), len(g.vertices), len(g.edges))
            try: path = g.dijkstra(tuple(start), tuple(goal))
            except:
                # samples_cfree = list(filter(lambda sample: not environment.check_robot_collision(self.to_pose_matrix(sample)), self.sample_config_space()))
                # samples = np.vstack((samples, samples_cfree))
                # self.add_edges_to_graph(samples,samples_cfree, environment, g)
                samples = self.sample_config_space(n=60)
                samples = np.vstack((samples, [start]))
                samples = np.vstack((samples, [goal]))
                samples = list(filter(lambda sample: not environment.check_robot_collision(self.to_pose_matrix(sample)), samples))
                g = self.build_graph(samples, samples, environment)

        plan = []
        # print("path length " + str(len(path)))
        # print(path)
        for i,j in zip(range(len(path) - 1), range(1, len(path))):
            _, sequence = self.path_check(np.array(path[i]), np.array(path[j]), environment)
            plan += sequence
        print(len(list(filter(lambda x: x[0] < self.limits[0][0] or x[0] > self.limits[0][1] or x[1] < self.limits[1][0] or x[1] > self.limits[1][1] or x[2] < self.limits[2][0] or x[2] > self.limits[2][1], plan))))
        # print(len(plan))
        return([self.to_pose_matrix(node) for node in plan])
        # return []
    
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

