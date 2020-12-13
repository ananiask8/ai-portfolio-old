#!/usr/bin/env python3
 
import numpy as np
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og
 
def plot_points(points, specs='r'):
    """
    method to plot the points
 
    Parameters
    ----------
    points: list(float, float)
        list of the pint coordinates
    """
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]
    plt.plot(x_val, y_val, specs)
    plt.draw()
 
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
    return (state[0] >= state[1])
 
def plan(start, goal):
    #set dimensions of the problem to be solved
    stateSpace = ob.RealVectorStateSpace()  #set state space
    stateSpace.addDimension(0.0, 10.0) #set width
    stateSpace.addDimension(0.0, 10.0) #set height
 
    #create a simple setup object
    task = og.SimpleSetup(stateSpace)
 
    #set methods for collision detection and resolution of the checking 
    task.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    task.getSpaceInformation().setStateValidityCheckingResolution(0.001)
 
    #setting start and goal positions
    start_pose = ob.State(task.getStateSpace())   
    goal_pose = ob.State(task.getStateSpace())
    start_pose()[0] = start[0]
    start_pose()[1] = start[1]
    goal_pose()[0] = goal[0]
    goal_pose()[1] = goal[1]
    task.setStartAndGoalStates(start_pose, goal_pose)
 
    #setting particular planner from the supported planners
    info = task.getSpaceInformation()
    planner = og.RRT(info)   # RRT, RRTConnect, FMT, ...
    task.setPlanner(planner)
 
    #find the solution
    solution = task.solve()

    #simplify the solution
    if solution:
       task.simplifySolution()
 
    #retrieve found path
    plan = task.getSolutionPath()
 
    #extract path and plot it
    path = []
    for i in range(plan.getStateCount()):
        path.append((plan.getState(i)[0], plan.getState(i)[1]))
 
    plot_points(path,'r')
    plt.show() 
 
if __name__ == "__main__":
    plan([0, 0], [10, 10])