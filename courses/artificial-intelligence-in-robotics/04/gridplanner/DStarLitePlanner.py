#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np

import collections
import heapq

import matplotlib.pyplot as plt

import GridMap as gmap
from GridPlanner import GridPlanner

class DStarLitePlanner(GridPlanner):

    def __init__(self):
        self.first_run = True
        self.grid = None
        self.U = None
        
    def getRHSvalue(self, coord=None):
        """
        method to get the RHS values of the D* lite grid
        
        Parameters
        ----------
        coord: (int,int)
            if given, the function returns the RHS value only at the given point, otherwise it returns all the RHS values

        Returns
        -------
        np.array(float) 
            RHS values
        """
        ret = None
        if not coord == None:
            (x, y) = coord
            ret = self.grid["rhs"][x][y] 
        else:
            ret = self.grid["rhs"]
        return ret

    def getGvalue(self, coord = None):
        """
        method to get the G values of the D* lite grid
        
        Parameters
        ----------
        coord: (int,int)
            if given, the function returns the G value only at the given point, otherwise it returns all the G values

        Returns
        -------
        np.array(float) 
            G values
        """
        ret = None
        if not coord == None:
            (x, y) = coord
            ret = self.grid["g"][x][y] 
        else:
            ret = self.grid["g"]
        return ret
                
    def setRHSvalue(self, coord, value):
        """
        method to set the RHS value of the D* lite grid

        Parameters
        ----------
        coord: (int,int)
            coordinates of the cell to be updated
        value: float
            update value
        """
        (x, y) = coord
        self.grid["rhs"][x][y] = value
                
    def setGvalue(self, coord, value):
        """
        method to set the G value of the D* lite grid

        Parameters
        ----------
        coord: (int,int)
            coordinates of the cell to be updated
        value: float
            update value
        """
        (x, y) = coord
        self.grid["g"][x][y] = value


    def calculate_key(self, coord, goal, gridmap):
        """
        method to calculate the priority queue key

        Parameters
        ----------
        coord: (int, int)
            cell to calculate key for
        goal: (int, int)
            goal location
        
        Returns
        -------
        (float, float)
            major and minor key
        """
        major_key = min(self.getGvalue(coord), self.getRHSvalue(coord)) + gridmap.cost_manhattan(coord, goal)
        minor_key = min(self.getGvalue(coord), self.getRHSvalue(coord))

        return (major_key, minor_key)

    def compute_shortest_path(self, gridmap, start, goal):
        """
        Function to compute the shortest path

        Parameters
        ----------
        gridmap:GridMap
            map of the environment
        start: (int, int)
            start position
        goal: (int, int)
            goal position
        """
        # while not self.U.empty():
        while not self.U.empty() and (self.U.top_key() < self.calculate_key(start, start, gridmap) or self.getRHSvalue(start) != self.getGvalue(start)):
            u = self.U.get()
            if self.getGvalue(u) > self.getRHSvalue(u):
                self.setGvalue(u, self.getRHSvalue(u))
                for s in gridmap.neighbors4(u):
                    self.update_vertex(gridmap, s, start, goal)
            else:
                # print("SET G", u)
                self.setGvalue(u, np.inf)
                for s in gridmap.neighbors4(u) + list([u]):
                    self.update_vertex(gridmap, s, start, goal)
        # print(self.U.empty(), self.U.top_key() <= self.calculate_key(start, goal, gridmap), self.getRHSvalue(start) != self.getGvalue(start))
        # print(self.U.top(), self.U.top_key(), self.calculate_key(start, goal, gridmap))
        # print((6,5), self.calculate_key((6,5), goal, gridmap))
        # print(self.getRHSvalue(start),self.getGvalue(start))
        # print(self.U.elements)

    def update_vertex(self, gridmap, u, start, goal):
        """
        Function for map vertex updating

        Parameters
        ----------
        gridmap:GridMap
            map of the environment
        u: (int, int)
            currently processed position
        start: (int, int)
            start position
        goal: (int, int)
            goal position
        """
        if not gridmap.passable(u): return
        if u != goal:
            self.setRHSvalue(u, min([(gridmap.dist_euclidean(u, s) + self.getGvalue(s)) for s in gridmap.neighbors4(u)]))
        if self.U.contains(u):
            self.U.remove(u)
        if self.getGvalue(u) != self.getRHSvalue(u):
            self.U.put(u, self.calculate_key(u, start, gridmap))

    def reconstruct_path(self, gridmap, start, goal):
        """
        Function to reconstruct the path

        Parameters
        ----------
        gridmap:GridMap
            map of the environment
        u: (int, int)
            currently processed position
        start: (int, int)
            start position
        goal: (int, int)
            goal position
        
        Returns
        -------
        list(int, int)
            the path
        """
        curr = start
        path = []
        path.append(curr)
        while curr != goal:
            neighbors = gridmap.neighbors4(curr)
            curr = neighbors[0]
            for neighbor in neighbors:
                if self.getGvalue(curr) > self.getGvalue(neighbor):
                    curr = neighbor
            if curr in path: return None
            path.append(curr)
        return(path)

    def cells_to_update(self, gridmap, start, goal):
        cells = []
        for j in range(gridmap.height):
            for i in range(gridmap.width):
                u = (i, j)
                if self.getRHSvalue(u) < np.inf and not gridmap.passable(u):
                    self.setGvalue(u, np.inf)
                    self.setRHSvalue(u, np.inf)
                    cells.append(u)
                if self.getRHSvalue(u) == np.inf and self.getGvalue(u) == np.inf and gridmap.passable(u):
                    self.update_vertex(gridmap, u, start, goal)
                    cells.append(u)
        return cells

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
        if self.first_run:
            self.first_run = False

            #create the grid
            width = gridmap.width
            height = gridmap.height
            self.grid = np.zeros((width, height), dtype=[('rhs',np.dtype(float)),('g',np.dtype(float))])
            
            #fill it with the default values
            self.grid['rhs'] = np.inf
            self.grid['g'] = np.inf
            
            #create the priority queue
            self.U = self.PriorityQueue()
            
            #starting point
            self.grid['rhs'][goal[0]][goal[1]] = 0
            self.U.put(goal, self.calculate_key(goal, start, gridmap))

            #compute shortest path
            self.compute_shortest_path(gridmap, start, goal)
            
            #reconstruct path
            return self.reconstruct_path(gridmap, start, goal)
        else:
            #find the change to the map
            changed_cells = self.cells_to_update(gridmap, start, goal)
            if len(changed_cells) > 0:
                for u in changed_cells:
                    # update the rhs and g values of the changed cell and its neighbors
                    for s in gridmap.neighbors4(u):
                        # print(s)
                        self.update_vertex(gridmap, s, start, goal)

                    # update everything in the queue
                    for j in range(gridmap.height):
                        for i in range(gridmap.width):
                            s = (i, j)
                            if self.U.contains(s):
                                self.U.remove(s)
                                self.U.put(s, self.calculate_key(s, start, gridmap))
                #recompute the shortest path
                self.compute_shortest_path(gridmap, start, goal)
            #reconstruct path
            return self.reconstruct_path(gridmap, start, goal)



    ###############################
    #priority queue class
    ###############################
    class PriorityQueue:
        def __init__(self):
            self.elements = []
        
        def empty(self):
            return len(self.elements) == 0
        
        def put(self, item, priority):
            heapq.heappush(self.elements, (priority, item))
        
        def get(self):
            return heapq.heappop(self.elements)[1]
        
        def top(self):
            u = self.elements[0]
            return u[1]

        def top_key(self):
            u = self.elements[0]
            return u[0]
        
        def contains(self, element):
            ret = False
            for item in self.elements:
                if element == item[1]:
                    ret = True
                    break
            return ret
        
        def remove(self, element):
            i = 0
            for item in self.elements:
                if element == item[1]:
                    self.elements[i] = self.elements[-1]
                    self.elements.pop()
                    heapq.heapify(self.elements)
                    break
                i += 1
