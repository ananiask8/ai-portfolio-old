#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import scipy.ndimage as ndimg

import matplotlib.pyplot as plt
import csv


class GridMap:

    def __init__(self, width=0, height=0, scale=0.1):
        """
        Parameters
        ----------
        width : int, optional
            width of the grid map
        height : int, optional
            height of the grid map
        scale : double, optional
            the scale of the map 
        """
        if width > 0 and height > 0:
            self.grid = np.zeros((width, height), dtype=[('p',np.dtype(float)),('free',np.dtype(bool))])
            self.grid['p'] = 0.5    #set all probabilities to 0.5
            self.grid['free'] = True #set all cells to passable
            self.width = width
            self.height = height
            self.scale = scale
        else:
            self.grid = None
            self.width = None
            self.height = None
            self.scale = None

    def load_map(self, filename, scale=0.1):
        """ 
        Load map from file 
        
        Parameters
        ----------
        filename : str 
            path to the map file
        scale : double (optional)
            the scale of the map
        """
        grid = np.genfromtxt(filename, delimiter=',')
        grid = grid*0.95 + 0.025
        grid = np.flip(grid,0) #to cope with the vrep scene

        self.grid = np.zeros(grid.shape, dtype=[('p',np.dtype(float)),('free',np.dtype(bool))])
        self.grid['p'] = grid   #populate the occupancy grid
        self.grid['free'] = True    #assign passable and impassable cells
        self.grid['free'][grid > 0.5] = False

        self.width = self.grid.shape[0]
        self.height = self.grid.shape[1]
        self.scale = scale

    def bresenham_line(self, start, goal, eps = 0, allow_free = True):
        "Bresenham's line algorithm"
        cells_to_update = []
        x0, y0 = start
        x1, y1 = goal
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while round(x + eps - x1) != 0 and round(x - eps - x1) != 0:
                cells_to_update.append((int(math.floor(x)), int(math.floor(y)), allow_free))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while round(y + eps - y1) != 0 and round(y - eps - y1) != 0:
                cells_to_update.append((int(math.floor(x)), int(math.floor(y)), allow_free))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy        
        cells_to_update.append((int(math.floor(x1)), int(math.floor(y1)), False))
        cells_to_update.append((int(math.floor(x1 + eps)), int(math.floor(y1)), False))
        cells_to_update.append((int(math.floor(x1 - eps)), int(math.floor(y1)), False))
        cells_to_update.append((int(math.floor(x1)), int(math.floor(y1 + eps)), False))
        cells_to_update.append((int(math.floor(x1)), int(math.floor(y1 - eps)), False))
        return cells_to_update

    def bayesian_update(self, point, p = 0.9):
        s_occupied = 0 if point[2] else p
        s_free = p if point[2] else 0
        p_z_occupied = (1 + s_occupied - s_free)/2.0
        p_z_free = 1 - p_z_occupied

        prior_occ = self.get_cell_p(point[:2])
        prior_free = 1 - prior_occ
        p_occ_z = prior_occ*p_z_occupied/(p_z_occupied*prior_occ + p_z_free*prior_free)
        self.set_cell_p(point[:2], p_occ_z)

    def fuse_laser_scan(self, pose, scan_x, scan_y):
        """
        Method to fuse the laser scanner data into the map

        Parameters
        ----------
        pose: (float, float, float)
            pose of the robot (x, y, orientation)
        scan_x: list(float)
        scan_y: list(float)
            relative x and y coordinates of the points in laser scan w.r.t. the current robot pose
        """
        if pose is None or len(scan_x) == 0 or len(scan_y) == 0: return

        # Offset to the robot's coordinates
        measures = np.array([np.array(scan_x), np.array(scan_y)])

        # Rotate measures
        theta = pose[2]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_measures = np.dot(rotation_matrix, measures)
        rotated_measures[0,:] += pose[0]
        rotated_measures[1,:] += pose[1]

        # Raytracing using Bresenham's line algorithm
        cells_to_update = []
        for i in range(len(scan_x)):
            # print(rotated_measures[:,i])
            # exit()
            cells_to_update += self.bresenham_line(np.divide(pose[:2], self.scale if self.scale is not None else 1), np.divide(rotated_measures[:,i], self.scale if self.scale is not None else 1), 1)
        
        # Update occupancy grid
        for p in cells_to_update:
            # print(pose, p)
            if self.in_bounds(p[:2]): self.bayesian_update(p)

    def cells_to_grow(self, p, d):
        x, y = p
        cells_to_update = []
        d = int(d)
        for j in range(d):
            for i in range(d):
                if i**2 + j**2 <= d**2:
                    cells_to_update.append((x+i, y+j, True))
                    cells_to_update.append((x+i, y-j, True))
                    cells_to_update.append((x-i, y+j, True))
                    cells_to_update.append((x-i, y-j, True))
        # for d in [(math.cos(2*math.pi/n*x)*math.sqrt(r),math.sin(2*math.pi/n*x)*math.sqrt(r)) for x in xrange(0,n+1)]:
        #     cells_to_update += self.bresenham_line(start, np.array(start) + np.array(d), 0, False)
        return cells_to_update

    def grow_obstacles(self, distance):
        """
        Method to blow the obstacles to prevent the robot from hitting the obstacles
        The method sets the 'free' flag for the map points that are closer to the obstacles than a given threshold distance

        Parameters
        ----------
        distance: float
            the distance for blowing the obstacles in world units
        """
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.passable((x, y)):
                    cells += self.cells_to_grow((x, y), np.divide(distance, self.scale))

        for xd,yd,free in cells:
            if self.in_bounds((xd, yd)): self.grid[xd,yd]['free'] = False

    def get_cell_p(self, coord):
        """
        Get content of the grid cell

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        float
            cell value or None if the cell is outside of the map
        """
        (x, y) = coord 
        ret = None
        if self.in_bounds(coord):
            ret = self.grid[x, y]['p']
        return ret

    def set_cell_p(self, coord, value):
        """
        Set content of the grid cell

        Parameters
        ----------
        coord : (int, int)
            map coordinate
        value : float
            cell value

        Returns
        -------
        bool
            True if the value has been set, False if the coordinates are outside of the grid map 
        """
        (x, y) = coord
        ret = self.in_bounds(coord)
        if ret:
            self.grid[x, y]['p'] = value
            if value > 0.5:
                self.grid[x, y]['free'] = False
            else:
                self.grid[x, y]['free'] = True

        return ret


    ##################################################
    # Planning helper functions
    ##################################################

    def in_bounds(self, coord):
        """
        Check the boundaries of the map

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        bool
            if the value is inside or outside the grid map dimensions
        """
        (x,y) = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, coord):
        """
        Check the passability of the given cell

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        bool
            if the grid map cell is occupied or not
        """
        ret = False
        if self.in_bounds(coord):
            (x,y) = coord
            ret = self.grid[x,y]['free']
        return ret

    def probably_passable(self, coord):
        ret = False
        if self.in_bounds(coord):
            (x,y) = coord
            ret = self.grid[x,y]['p'] > 0.5
        return ret


    def neighbors4(self, coord):
        """
        Returns coordinates of passable neighbors of the given cell in 4-neighborhood

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        list (int,int)
            list of neighbor coordinates 
        """
        (x,y) = coord
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.passable, results))
        return results

    def neighbors8(self, coord):
        """
        Returns coordinates of passable neighbors of the given cell in 8-neighborhood

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        list (int,int)
            list of neighbor coordinates 
        """
        (x,y) = coord
        results = [(x+1, y+1), (x+1, y), (x+1, y-1),
                   (x,   y+1),           (x,   y-1),
                   (x-1, y+1), (x-1, y), (x-1, y-1)]
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.passable, results))
        return results

    def dist_euclidean_squared(self, coord1, coord2):
        """
        Returns the direct squared euclidean distance between two points

        Parameters
        ----------
        coord1 : (int, int)
            map coordinate
        coord2 : (int, int)
            map coordinate

        Returns
        -------
        float
            squared euclidean distance between two coordinates 
        """
        (x1, y1) = coord1
        (x2, y2) = coord2
        return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)

    def dist_euclidean(self, coord1, coord2):
        """
        Returns the direct euclidean distance between two points

        Parameters
        ----------
        coord1 : (int, int)
            map coordinate
        coord2 : (int, int)
            map coordinate

        Returns
        -------
        float
            euclidean distance between two coordinates 
        """
        return math.sqrt(self.dist_euclidean_squared(coord1, coord2))

    def cost_manhattan(self, coord1, coord2):
        """
        Returns the direct manhattan distance between two points

        Parameters
        ----------
        coord1 : (int, int)
            map coordinate
        coord2 : (int, int)
            map coordinate

        Returns
        -------
        float
            manhattan distance between two coordinates 
        """
        (x1, y1) = coord1
        (x2, y2) = coord2
        return abs(x1 - x2) + abs(y1 - y2)

    ##############################################
    ## Helper functions for coordinate calculation
    ##############################################

    def world_to_map(self, coord):
        """
        function to return the map coordinates given the world coordinates

        Parameters
        ----------
        coord: list(float, float)
            list of world coordinates

        Returns
        -------
        list(int, int)
            list of map coordinates using proper scale and closest neighbor
        """
        cc = np.array(coord)/self.scale
        ci = cc.astype(int)
        if len(ci) > 2:
            return tuple(map(tuple, ci))
        else:
            return tuple(ci)

    def map_to_world(self, coord):
        """
        function to return the world coordinates given the map coordinates

        Parameters
        ----------
        coord: list(int, int)
            list of map coordinates

        Returns
        -------
        list(float, float)
            list of world coordinates
        """
        ci = np.array(coord).astype(float)
        cc = ci*self.scale
        if len(cc) > 2:
            return tuple(map(tuple, cc))
        else:
            return tuple(cc)

##############################################
## Helper functions to plot the map
##############################################
def plot_map(grid_map, cmap="Greys", clf = True, data='p'):
    """
    Method to plot the occupany grid map
    
    Parameters
    ----------
    grid_map : GridMap
        the GridMap object
    cmap : String(optional)
        the color map to be used
    clf : bool (optional)
        switch to clear or not to clear the drawing canvas
    data : String (optional)
        The grid map data to be displayed, 'p' for occupancy grid, 'free' for obstacle map
    """
    if clf:  #clear the canvas
        plt.clf()
    if data == 'p':
        plt.imshow(np.transpose(grid_map.grid['p']), cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        # plt.imshow(grid_map.grid['p'], cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    elif data == 'free':
        grid = np.transpose(grid_map.grid['free'])
        # grid = grid_map.grid['free']
        grid = np.array(grid, dtype=int)
        plt.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    else:
        print('Unknown data entry')
    plt.draw()

def plot_path(path, color="red"):
    """
    Method to plot the path
    
    Parameters
    ----------
    path: list(int,int)
        the list of individual waypoints
    """
    x_val = [x[0] for x in path]
    y_val = [x[1] for x in path]
    plt.plot(x_val, y_val, 'o-', color=color)
    plt.draw()

def test_gridmap(g):
    # positions has [(x, y, theta)]
    # laser_x has [(x1, x2, x3, ..., xi, ..., xn)]
    # laser_y has [(y1, y2, y3, ..., yi, ..., yn)]
    # Measures gathered at instant i are (positions[i], laser_x[i], laser_y[i])

    f = open('maps/measures_1.txt', 'r')
    x = f.read().splitlines()
    positions = []
    laser_x = []
    laser_y = []
    for i in range(len(x)):
        if i % 3 == 0: positions.append(tuple([float(elem) for elem in x[i].split(", ")]))
        if i % 3 == 1: laser_x.append(tuple([float(elem) for elem in x[i].split(", ")]))
        if i % 3 == 2: laser_y.append(tuple([float(elem) for elem in x[i].split(", ")]))
    f.close()

    for i in range(len(positions)):
        pose, scan_x, scan_y = positions[i], laser_x[i], laser_y[i]
        g.fuse_laser_scan(pose, scan_x, scan_y)
    g.grow_obstacles(0.2)
    plot_map(g, data='free')
    plt.pause(0.1)
    input()

if __name__=="__main__":
    #new map
    g = GridMap(100,100,0.1)
    # g.load_map("maps/maze01.csv", 0.1)
    # plot_map(g)
    # plt.show()
    # input("Please enter something: ")
    
    # point = np.array((5,50))
    # print(g.get_cell_p(point))
    # print(g.set_cell_p(point, 0.9))
    # point += (1,1)
    # print(point)
    # print(g.neighbors8(point))

    test_gridmap(g)
