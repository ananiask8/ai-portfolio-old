#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np

import collections
import heapq

import matplotlib.pyplot as plt

import GridMap as gmap
import time

class AStarGraph(object):
    def __init__(self, gridmap):
        self.barriers = []
        self.width, self.height = np.shape(gridmap.grid)
        self.gridmap = gridmap
 
    def heuristic(self, start, goal):
        return self.gridmap.dist_euclidean(goal, start)
 
    def get_vertex_neighbours(self, pos):
        return self.gridmap.neighbors8(pos)
 
    def move_cost(self, start, goal):
        if self.gridmap.passable(goal) and self.gridmap.probably_passable(goal):
            return sys.maxsize
        return self.gridmap.dist_euclidean(goal, start)

class GridPlanner:

    def __init__(self):
        pass

    def AStarSearch(self, start, end, graph):
     
        G = {} #Actual movement cost to each position from the start position
        F = {} #Estimated movement cost of start to end going via this position
     
        #Initialize starting values
        G[start] = 0 
        F[start] = graph.heuristic(start, end)
     
        closedVertices = set()
        openVertices = set([start])
        cameFrom = {}
     
        while len(openVertices) > 0:
            #Get the vertex in the open list with the lowest F score
            current = None
            currentFscore = None
            for pos in openVertices:
                if current is None or F[pos] < currentFscore:
                    currentFscore = F[pos]
                    current = pos
     
            #Check if we have reached the goal
            if current == end:
                #Retrace our route backward
                path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    path.append(current)
                path.reverse()
                return path
     
            #Mark the current vertex as closed
            openVertices.remove(current)
            closedVertices.add(current)
     
            #Update scores for vertices near the current position
            for neighbour in graph.get_vertex_neighbours(current):
                if neighbour in closedVertices: 
                    continue #We have already processed this node exhaustively
                candidateG = G[current] + graph.move_cost(current, neighbour)
     
                if neighbour not in openVertices:
                    openVertices.add(neighbour) #Discovered a new vertex
                elif candidateG >= G[neighbour]:
                    continue #This G score is worse than previously found
     
                #Adopt this G score
                cameFrom[neighbour] = current
                G[neighbour] = candidateG
                H = graph.heuristic(neighbour, end)
                F[neighbour] = G[neighbour] + H
     
        return None

    def plan(self, gridmap, start, goal):
        """
        Method to plan the path

        Parameters
        ----------
        gridmap: GridMap
            gridmap of the environment
        start: (int,int)
            start coordinates
        goal:(int,int)
            goal coordinates

        Returns
        -------
        list(int,int)
            the path between the start and the goal if there is one, None if there is no path
        """
        #TODO: Task03 - Implement a grid-based planner
        return self.AStarSearch(start, goal, AStarGraph(gridmap))

    def simplify_path(self, gridmap, path):
        """
        Method to implify the found path

        Parameters
        ----------
        gridmap: GridMap
            gridmap of the environment
        path: list(int,int)
            the path found by the planner

        Returns
        -------
        list(int,int)
            the simplified
        """
        #TODO: Task03 - Implement path simplification
        smooth_path = [path[0]]
        i = 0
        while i + 1 < len(path):
            for j in range(i + 1, len(path) + 1):
                if j >= len(path) or not all([gridmap.grid[(x,y)]['free'] for x,y,free in gridmap.bresenham_line(path[i], path[j])]):
                    smooth_path.append(path[j - 1])
                    break
            i = j - 1

        return smooth_path
