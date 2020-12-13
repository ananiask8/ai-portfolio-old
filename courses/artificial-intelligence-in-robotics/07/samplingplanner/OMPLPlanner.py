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
from ompl import base as ob
from ompl import geometric as og
from time import time

class OMPLPlanner:

    def __init__(self, limits, max_step_fraction_of_largest_dimension = 1/256.0, max_axis_orientation_change = np.pi/128.):
        """
        Parameters
        ----------
        limits: list((float, float))
            translation limits in individual axes 
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

    def to_pose_matrix(self, v):
        R = self.rotation_matrix(v[5],v[4],v[3])
        T = v[0:3]
        P = np.hstack((R,T.reshape((3,1))))
        P = np.vstack((P,[0,0,0,1]))

        return P

    def replan(self, environment, start, goal):
        def isStateValid(state):
            """
            method for collision detection checking
         
            Parameters
            ----------
            state : list(float)
                configuration to be checked for collision
         
            Returns
            -------
            bool
                False if there is collision, True otherwise
            """
            s = np.array([])
            for axis in range(len(self.limits)):
                s = np.hstack((s, state[axis]))

            return not environment.check_robot_collision(self.to_pose_matrix(s))

        resolution = min(self.max_translation[0], self.max_translation[-1])
        stateSpace = ob.RealVectorStateSpace()  #set state space
        stateSpace.setLongestValidSegmentFraction(1e-12)
        stateSpace.setValidSegmentCountFactor(8)
        for limit in self.limits:
            #set dimensions of the problem to be solved
            stateSpace.addDimension(limit[0], limit[1]) #set width
        #create a simple setup object
        task = og.SimpleSetup(stateSpace)

        #set methods for collision detection and resolution of the checking 
        task.getSpaceInformation().setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
        task.getSpaceInformation().setStateValidityCheckingResolution(resolution)

        #setting start and goal positions
        start_pose = ob.State(task.getStateSpace())
        goal_pose = ob.State(task.getStateSpace())
        for i in range(len(start)):
            start_pose()[i] = float(start[i])
            goal_pose()[i] = float(goal[i])
        task.setStartAndGoalStates(start_pose, goal_pose)

        #setting particular planner from the supported planners
        info = task.getSpaceInformation()
        planner = og.InformedRRTstar(info)   # RRT, RRTConnect, FMT, ...
        task.setPlanner(planner)

        #simplify the solution
        # ps = og.PathSimplifier(task.simplifySolution())

        #find the solution
        solution = task.solve(3600)
        if not task.haveSolutionPath(): return []
        print(task.haveExactSolutionPath())

        # #retrieve found path
        path = task.getSolutionPath()
        path.interpolate()
        # path.subdivide()
        # path.interpolate(int(path.length()/len(path.getStates())/resolution))

        # ps.smoothBSpline(path, maxSteps=100, minChange=(0.2*resolution))
        # i = 0
        # while not path.check():
        #     i += 1
        #     if i > 10: break
        #     # path.interpolate(100)
        #     print(path.check(), len(path.getStates()))
        #     try:
        #         # ps.reduceVertices(path)
        #         path.checkAndRepair()
        #     except: pass
        #     ps.reduceVertices(path)
        #     path.interpolate(int(path.length()/resolution))
        #     # path.subdivide()
        #     ps.smoothBSpline(path, maxSteps=100, minChange=(0.2*resolution))

        plan = []
        for configuration in path.getStates():
            s = np.array([])
            for axis in range(len(self.limits)):
                s = np.hstack((s, configuration[axis]))
            plan.append(s)

        return plan

    def get_plan_until_collision(self, environment, plan):
        colliding_idx = len(plan)
        for i in range(len(plan)):
            node = plan[i]
            if environment.check_robot_collision(self.to_pose_matrix(node)):
                colliding_idx = i
                break
                
        return plan if colliding_idx == len(plan) else plan[:max(colliding_idx-1, 1)]

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
        t1 = time()
        plan = [start]
        i = 0
        while not np.allclose(plan[-1], goal):
            i += 1
            if i >= 10: break
            partial_plan = self.replan(environment, plan[-1], goal)
            if len(partial_plan) > 0:
                plan = plan[:-1] + partial_plan
                plan = self.get_plan_until_collision(environment, plan)
                print(len(plan))
        t2  = time()
        print(t2-t1)
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

