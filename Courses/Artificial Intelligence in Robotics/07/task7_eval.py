#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

sys.path.append('environment')
sys.path.append('samplingplanner')
 
import Environment as env
import OMPLPlanner as op


if __name__ == "__main__":
    #define the planning scenarios
    #scenario name
    #start configuration
    #goal configuration
    #limits for individual DOFs
    scenarios = [("environments/maze1", (2,2,0,0,0,math.pi/4), (35,35,0,0,0,math.pi/2), [(0,40), (0,40), (0,0), (0,0), (0,0), (0,2*math.pi)]),
                 ("environments/maze2", (2,2,0,0,0,math.pi/4), (35,26,0,0,0,3*math.pi/2), [(0,40), (0,40), (0,0), (0,0), (0,0), (0,2*math.pi)]),
                 ("environments/maze3", (2,2,0,0,0,0), (25,36,0,0,0,0), [(0,40), (0,40), (0,0), (0,0), (0,0), (0,2*math.pi)]),
                 ("environments/maze4", (2,2,0,0,0,0), (27,36,0,0,0,math.pi/2), [(0,40), (0,40), (0,0), (0,0), (0,0), (0,2*math.pi)]),
                 ("environments/cubes", (-1,-1,1,0,0,0), (6,6,-6,math.pi,0,math.pi/2), [(-2,7), (-2,7), (-7,2), (0,2*math.pi), (0,2*math.pi), (0,2*math.pi)]),
                 ("environments/cubes2", (2,2,2,0,0,0), (19,19,19,0,0,0), [(-10,30), (-10,30), (-10,30), (0,2*math.pi), (0,2*math.pi), (0,2*math.pi)]),
("environments/alpha_puzzle", (0,5,0,0,0,0),(25,25,25,0,0,0), [(-40,70),(-40,70),(-40,70),(0,2*math.pi),(0,2*math.pi),(0,2*math.pi)])]

    # scenarios = [("environments/maze3", (2,2,0,0,0,0), (25,36,0,0,0,0), [(0,40), (0,40), (0,0), (0,0), (0,0), (0,2*math.pi)])]
    scenarios = [("environments/alpha_puzzle", (0,5,0,0,0,0),(25,25,25,0,0,0), [(-40,70),(-40,70),(-40,70),(0,2*math.pi),(0,2*math.pi),(0,2*math.pi)])]
    #enable dynamic drawing in matplotlib
    plt.ion()

    ########################################
    ## EVALUATION OF THE OMPL PLANNER
    ########################################
    for scenario in scenarios:
        name = scenario[0]
        start = np.asarray(scenario[1])
        goal = np.asarray(scenario[2])
        limits = scenario[3]
        
        print("processing scenario: " + name)

        #initiate environment and robot meshes
        environment = env.Environment()
        environment.load_environment(name)

        #instantiate the planner
        planner = op.OMPLPlanner(limits)
    
        #plan the path through the environment
        path = planner.plan(environment, start, goal)

        #plot the path step by step
        ax = None
        input()
        for Pose in path:
            ax = environment.plot_environment(Pose, ax=ax, limits=limits)
            plt.pause(0.01)
