# -*- coding: utf-8 -*-
"""

Multi-Goal Motion Planner (MGMP)

@author: P.Vana & P.Cizek
"""
import sys
import os
import numpy as np
from collections import defaultdict
import itertools
import copy

import math

import GridMap as gmap
import GridPlanner as gplanner

import matplotlib.pyplot as plt

from invoke_LKH import *


class MGMP_Solver:
    def __init__(self):
        pass

    def get_array_of_coordinates(self, n, m):
        coords = []
        for i in range(n):
            for j in range(m):
                coords.append((i, j))
        return coords

    def prepare_weight_matrix(self, gridmap, goals):
        w = defaultdict(dict)
        for coord in list(filter(lambda coord: gridmap.passable(coord), self.get_array_of_coordinates(gridmap.width, gridmap.height))):
            if gridmap.passable(coord):
                for neighbour in gridmap.neighbors4(coord):
                    w[coord][neighbour] = gridmap.dist_euclidean(coord, neighbour)
                    # print(coord, neighbour, w[coord][neighbour])
        return w

    def floyd_warshall(self, gridmap, w):
        coords = list(filter(lambda coord: gridmap.passable(coord), self.get_array_of_coordinates(gridmap.width, gridmap.height)))
        dist, path = defaultdict(dict), defaultdict(dict)
        for coord1 in coords:
            for coord2 in coords:
                dist[coord1][coord1] = 0
                dist[coord2][coord2] = 0
                dist[coord1][coord2] = np.inf
                if coord1 in w and coord2 in w[coord1]:
                    dist[coord1][coord2] = w[coord1][coord2]
                    path[coord1][coord2] = [coord1, coord2]

        for k in coords:
            for i in coords:
                for j in coords:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        path[i][j] = path[i][k] + path[k][j]

        return dist, path

    # compute the shortest sequence based on the distance matrix (self.distances)
    def compute_TSP_sequence(self, start_idx = None, end_idx = None):
        n = len(self.distances)
        
        if start_idx is not None and end_idx is not None:
            M = n * np.max(np.max(self.distances))
            for i in range(n):
                self.distances[i, start_idx] = M
                self.distances[end_idx, i] = M
            self.distances[end_idx, start_idx] = 0
        
        fname_tsp = "problem"
        user_comment = "a comment by the user"
        writeTSPLIBfile_FE(fname_tsp, self.distances,user_comment)
        run_LKHsolver_cmd(fname_tsp)
        sequence = read_LKHresult_cmd(fname_tsp)
        return sequence
    
    # compute the shortest tour based on the distance matrix (self.distances)
    def compute_TSP_tour(self, start_idx = None, end_idx = None):
        n = len(self.distances)
        sequence = self.compute_TSP_sequence(start_idx=start_idx, end_idx=end_idx)
        
        visited = set([])
        if start_idx != None and end_idx != None:
            path = []
            for a in range(0,n):
                b = (a+1) % n
                a_idx = sequence[a]
                b_idx = sequence[b]
                if b_idx not in visited:
                    actual_path = self.paths[(a_idx,b_idx)]
                    path = path + actual_path
                    visited.add(a_idx)
                    visited.add(b_idx)
        else:
            path = []
            for a in range(0,n):
                b = (a+1) % n
                a_idx = sequence[a]
                b_idx = sequence[b]
                actual_path = self.paths[(a_idx,b_idx)]
                path = path + actual_path
        
        return path
        
    
    def plan_tour_etsp(self, gridmap, goals, start_idx = None, end_idx = None):
        n = len(goals)
        self.distances = np.zeros((n,n))	
        self.paths = {}

        planner = gplanner.GridPlanner()

        # find path between each pair of goals (a,b)
        for a in range(0,n): 
            for b in range(0,n):
                g1 = goals[a]
                g2 = goals[b]
                if a != b:
                    # store distance
                    self.distances[a][b] = gridmap.dist_euclidean(g1, g2) 
                    # store path
                    self.paths[(a,b)] = [g1, g2]

        return self.compute_TSP_tour(start_idx, end_idx)
    
    def plan_tour_obstacles(self, gridmap, goals, start_idx = None, end_idx = None):	
        n = len(goals)
        self.distances = np.zeros((n,n))	
        self.paths = {}

        planner = gplanner.GridPlanner()

        # find path between each pair of goals (a,b)
        for a in range(0,n): 
            print('Shortest paths from city {0}'.format(a))
            for b in range(0,n):
                if a != b:
                    g1 = goals[a]
                    g2 = goals[b]
                
                    # find the shortest path
                    sh_path = planner.plan(gridmap, g1[:2], g2[:2])
                    # store distance
                    dist = 0
                    last = sh_path[0]
                    for act in sh_path:
                        dist += gridmap.dist_euclidean(last, act)
                        last = act 
                    self.distances[a][b] = dist
                    # store path
                    self.paths[(a,b)] = sh_path

        return self.compute_TSP_tour(start_idx, end_idx)
    
    def plan_tour_obstacles_Floyd_Warshall(self, gridmap, goals, start_idx = None, end_idx = None):	
        n = len(goals)
        self.distances = np.zeros((n,n))    
        self.paths = {}

        planner = gplanner.GridPlanner()
        w = self.prepare_weight_matrix(gridmap, goals)
        distances, paths = self.floyd_warshall(gridmap, w)

        for a in range(n): 
            for b in range(n):
                g1 = goals[a]
                g2 = goals[b]
                if a != b:
                    self.distances[a][b] = distances[g1][g2]
                    self.paths[(a,b)] = paths[g1][g2]

        # print(goals[start_idx], goals[end_idx])
        return self.compute_TSP_tour(start_idx, end_idx)
    
    def get_path(self, x, y, gridmap, planner):
        path = planner.plan(gridmap, x, y, neigh = 'N8')
        dist = 0
        last = path[0]
        for act in path:
            dist += gridmap.dist_euclidean(last, act)
            last = act
        return dist, path

    def get_all_neighbourhood_distances(self, gridmap, flat_neighbours, planner):
        neighbour_distances = defaultdict(dict)
        paths = defaultdict(dict)
        for a in flat_neighbours:
            for b in flat_neighbours:
                neighbour_distances[a][b], paths[a][b] = self.get_path(a, b, gridmap, planner)
        return neighbour_distances, paths

    def get_neighbourhood_distances(self, gridmap, neighbors, sequence, planner):
        n = len(sequence)
        neighbour_distances = defaultdict(dict)
        for k in range(n):
            i = sequence[k]
            j = sequence[np.mod(k+1, n)]
            for a in neighbors[i]:
                for b in neighbors[j]:
                    neighbour_distances[a][b], _ = self.get_path(a, b, gridmap, planner)
        return neighbour_distances

    def select_neighbours(self, sequence, neighbour_distances, neighbors):
        n = len(sequence)
        sample = [0]*n
        best = sum([neighbour_distances[neighbors[sequence[i]][0]][neighbors[sequence[(i+1)%n]][0]] for i in range(n)])
        improving = True
        while improving:
            improving = False
            for j in range(n):
                i = (j-1) % n
                k = (j+1) % n
                prev_idx, current_idx, next_idx = sequence[i], sequence[j], sequence[k]
                prev_r, current_r, next_r = neighbors[prev_idx], neighbors[current_idx], neighbors[next_idx]
                prev_v_idx, current_v_idx, next_v_idx = sample[i], sample[j], sample[k]
                prev_v, current_v, next_v = prev_r[prev_v_idx], current_r[current_v_idx], next_r[next_v_idx]
                current_add_component = neighbour_distances[prev_v][current_v] + neighbour_distances[current_v][next_v]

                m = len(current_r)
                for new_v_idx in range(m):
                    new_v = current_r[new_v_idx]
                    new_add_component = neighbour_distances[prev_v][new_v] + neighbour_distances[new_v][next_v]
                    if best - current_add_component + new_add_component < best:
                        improving = True
                        best += new_add_component - current_add_component
                        sample[j] = new_v_idx
                        break # necessary or save new add component to current add component
        return best, [neighbors[sequence[i]][sample[i]] for i in range(n)]

    def plan_tour_obstacles_neighborhood(self, gridmap, goals, neighbors, start_idx = None, end_idx = None):    
        n = len(goals)
        self.distances = np.zeros((n,n))    
        self.paths = {}
 
        planner = gplanner.GridPlanner()
 
        # compute sequence based on the non-collision Euclidean distances
        for a in range(0,n): 
            print('Shortest paths from city {0}'.format(a))
            for b in range(0,n):
                if a != b:
                    g1 = goals[a]
                    g2 = goals[b]
                 
                    # find the shortest path
                    sh_path = planner.plan(gridmap, g1, g2, neigh = 'N8')
                    # store distance
                    dist = 0
                    last = sh_path[0]
                    for act in sh_path:
                        dist += gridmap.dist_euclidean(last, act)
                        last = act 
                    self.distances[a][b] = dist
                    # store path
                    self.paths[(a,b)] = sh_path
 
        sequence = self.compute_TSP_sequence(start_idx, end_idx)
 
        neighbour_distances = self.get_neighbourhood_distances(gridmap, neighbors, sequence, planner)
        # neighbour_distances = {(30, 80): {(10, 39): 49.284271247461874, (13, 14): 73.04163056034257, (34, 46): 35.656854249492376, (23, 9): 73.89949493661165, (11, 16): 71.87005768508875, (16, 45): 40.79898987322332, (41, 16): 68.55634918610401, (11, 15): 72.87005768508875, (13, 13): 74.04163056034257, (24, 47): 35.48528137423857}, (33, 106): {(10, 39): 76.52691193458112, (13, 14): 100.28427124746185, (34, 46): 60.41421356237309, (23, 9): 101.14213562373092, (11, 16): 99.11269837220803, (16, 45): 68.04163056034257, (41, 16): 93.31370849898474, (11, 15): 100.11269837220803, (13, 13): 101.28427124746185, (24, 47): 62.72792206135783}, (17, 76): {(10, 39): 39.89949493661165, (13, 14): 63.65685424949237, (34, 46): 37.04163056034262, (23, 9): 69.48528137423855, (11, 16): 62.485281374238554, (16, 45): 31.414213562373096, (41, 16): 69.94112549695421, (11, 15): 63.485281374238554, (13, 13): 64.65685424949237, (24, 47): 31.89949493661167}, (36, 97): {(10, 39): 68.7695526217004, (13, 14): 92.52691193458112, (34, 46): 51.828427124746185, (23, 9): 93.3847763108502, (11, 16): 91.3553390593273, (16, 45): 60.284271247461845, (41, 16): 83.07106781186546, (11, 15): 92.3553390593273, (13, 13): 93.52691193458112, (24, 47): 54.97056274847711}, (21, 76): {(10, 39): 41.55634918610403, (13, 14): 65.31370849898474, (34, 46): 35.384776310850235, (23, 9): 67.82842712474618, (11, 16): 64.14213562373092, (16, 45): 33.071067811865476, (41, 16): 68.28427124746185, (11, 15): 65.14213562373092, (13, 13): 66.31370849898474, (24, 47): 30.242640687119287}, (36, 96): {(10, 39): 67.7695526217004, (13, 14): 91.52691193458112, (34, 46): 50.828427124746185, (23, 9): 92.3847763108502, (11, 16): 90.3553390593273, (16, 45): 59.284271247461845, (41, 16): 82.07106781186546, (11, 15): 91.3553390593273, (13, 13): 92.52691193458112, (24, 47): 53.97056274847711}, (36, 92): {(10, 39): 63.76955262170041, (13, 14): 87.52691193458112, (34, 46): 46.828427124746185, (23, 9): 88.3847763108502, (11, 16): 86.3553390593273, (16, 45): 55.28427124746186, (41, 16): 78.07106781186546, (11, 15): 87.3553390593273, (13, 13): 88.52691193458112, (24, 47): 49.97056274847711}, (17, 115): {(10, 39): 78.89949493661165, (13, 14): 102.65685424949237, (34, 46): 76.04163056034257, (23, 9): 108.48528137423855, (11, 16): 101.48528137423855, (16, 45): 70.41421356237309, (41, 16): 108.94112549695421, (11, 15): 102.48528137423855, (13, 13): 103.65685424949237, (24, 47): 70.89949493661165}, (27, 78): {(10, 39): 46.0416305603426, (13, 14): 69.79898987322329, (34, 46): 34.89949493661166, (23, 9): 70.65685424949237, (11, 16): 68.62741699796948, (16, 45): 37.55634918610404, (41, 16): 67.79898987322329, (11, 15): 69.62741699796948, (13, 13): 70.79898987322329, (24, 47): 32.242640687119284}, (17, 114): {(10, 39): 77.89949493661165, (13, 14): 101.65685424949237, (34, 46): 75.04163056034257, (23, 9): 107.48528137423855, (11, 16): 100.48528137423855, (16, 45): 69.41421356237309, (41, 16): 107.94112549695421, (11, 15): 101.48528137423855, (13, 13): 102.65685424949237, (24, 47): 69.89949493661165}, (10, 39): {(151, 158): 196.14927829866758, (183, 181): 253.49242404917516, (178, 151): 221.42135623730962, (172, 187): 254.93607486307098, (150, 160): 197.73506473629448, (187, 174): 248.14927829866758, (173, 149): 217.3502884254441, (181, 153): 224.66399692442894, (184, 179): 251.90663761154826, (187, 162): 309.66399692442894}, (13, 14): {(151, 158): 204.67619023324886, (183, 181): 262.01933598375643, (178, 151): 229.9482681718909, (172, 187): 263.46298679765226, (150, 160): 206.26197667087575, (187, 174): 256.67619023324886, (173, 149): 225.87720036002537, (181, 153): 233.19090885901022, (184, 179): 260.43354954612954, (187, 162): 318.1909088590102}, (34, 46): {(151, 158): 169.24978336205578, (183, 181): 226.59292911256352, (178, 151): 194.521861300698, (172, 187): 228.03657992645935, (150, 160): 170.83556979968267, (187, 174): 221.24978336205595, (173, 149): 190.45079348883246, (181, 153): 197.7645019878173, (184, 179): 225.00714267493663, (187, 162): 282.7645019878173}, (23, 9): {(151, 158): 205.5340546095179, (183, 181): 262.8772003600255, (178, 151): 230.80613254815995, (172, 187): 264.3208511739213, (150, 160): 207.1198410471448, (187, 174): 257.5340546095179, (173, 149): 226.73506473629442, (181, 153): 234.04877323527927, (184, 179): 261.2914139223986, (187, 162): 319.04877323527927}, (11, 16): {(151, 158): 204.67619023324886, (183, 181): 262.01933598375643, (178, 151): 229.9482681718909, (172, 187): 263.46298679765226, (150, 160): 206.26197667087575, (187, 174): 256.67619023324886, (173, 149): 225.87720036002537, (181, 153): 233.19090885901022, (184, 179): 260.43354954612954, (187, 162): 318.1909088590102}, (16, 45): {(151, 158): 187.663996924429, (183, 181): 245.00714267493657, (178, 151): 212.93607486307104, (172, 187): 246.4507934888324, (150, 160): 189.2497833620559, (187, 174): 239.663996924429, (173, 149): 208.8650070512055, (181, 153): 216.17871555019036, (184, 179): 243.42135623730968, (187, 162): 301.17871555019036}, (41, 16): {(151, 158): 191.0782104868021, (183, 181): 248.42135623730968, (178, 151): 216.35028842544415, (172, 187): 249.8650070512055, (150, 160): 192.663996924429, (187, 174): 243.0782104868021, (173, 149): 212.27922061357862, (181, 153): 219.59292911256347, (184, 179): 246.8355697996828, (187, 162): 304.59292911256347}, (11, 15): {(151, 158): 205.09040379562197, (183, 181): 262.43354954612954, (178, 151): 230.362481734264, (172, 187): 263.87720036002537, (150, 160): 206.67619023324886, (187, 174): 257.09040379562197, (173, 149): 226.29141392239848, (181, 153): 233.60512242138333, (184, 179): 260.84776310850265, (187, 162): 318.6051224213833}, (13, 13): {(151, 158): 205.67619023324886, (183, 181): 263.01933598375643, (178, 151): 230.9482681718909, (172, 187): 264.46298679765226, (150, 160): 207.26197667087575, (187, 174): 257.67619023324886, (173, 149): 226.87720036002537, (181, 153): 234.19090885901022, (184, 179): 261.43354954612954, (187, 162): 319.1909088590102}, (24, 47): {(151, 158): 178.83556979968276, (183, 181): 236.17871555019042, (178, 151): 204.10764773832489, (172, 187): 237.62236636408625, (150, 160): 180.42135623730965, (187, 174): 230.83556979968284, (173, 149): 200.03657992645935, (181, 153): 207.3502884254442, (184, 179): 234.59292911256352, (187, 162): 292.3502884254442}, (151, 158): {(270, 236): 188.0538238691624, (251, 229): 171.95331880577413, (264, 231): 184.12489168102792, (250, 229): 170.95331880577413, (247, 230): 167.53910524340102, (267, 264): 193.33809511662452, (242, 233): 161.2964645562817, (243, 264): 169.33809511662452, (237, 255): 159.61017305526656, (266, 264): 192.33809511662452}, (183, 181): {(270, 236): 126.56854249492372, (251, 229): 110.46803743153536, (264, 231): 122.63961030678918, (250, 229): 109.46803743153536, (247, 230): 106.05382386916227, (267, 264): 131.8528137423856, (242, 233): 99.811183182043, (243, 264): 107.85281374238556, (237, 255): 98.12489168102773, (266, 264): 130.8528137423856}, (178, 151): {(270, 236): 158.63961030678922, (251, 229): 142.53910524340094, (264, 231): 154.71067811865476, (250, 229): 141.53910524340094, (247, 230): 138.1248916810278, (267, 264): 163.92388155425135, (242, 233): 131.88225099390846, (243, 264): 139.92388155425112, (237, 255): 130.1959594928932, (266, 264): 162.92388155425135}, (172, 187): {(270, 236): 125.12489168102778, (251, 229): 109.02438661763942, (264, 231): 121.19595949289324, (250, 229): 108.02438661763942, (247, 230): 104.61017305526633, (267, 264): 130.40916292848965, (242, 233): 98.36753236814705, (243, 264): 106.40916292848962, (237, 255): 96.68124086713179, (266, 264): 129.40916292848962}, (150, 160): {(270, 236): 185.63961030678928, (251, 229): 169.53910524340102, (264, 231): 181.7106781186548, (250, 229): 168.53910524340102, (247, 230): 165.12489168102792, (267, 264): 190.9238815542514, (242, 233): 158.8822509939086, (243, 264): 166.9238815542514, (237, 255): 157.19595949289345, (266, 264): 189.9238815542514}, (187, 174): {(270, 236): 131.91168824543135, (251, 229): 115.81118318204297, (264, 231): 127.98275605729678, (250, 229): 114.81118318204297, (247, 230): 111.39696961966987, (267, 264): 137.19595949289325, (242, 233): 105.1543289325506, (243, 264): 113.19595949289317, (237, 255): 103.46803743153534, (266, 264): 136.19595949289325}, (173, 149): {(270, 236): 162.71067811865467, (251, 229): 146.61017305526642, (264, 231): 158.7817459305202, (250, 229): 145.61017305526642, (247, 230): 142.1959594928933, (267, 264): 167.9949493661168, (242, 233): 135.95331880577396, (243, 264): 143.99494936611663, (237, 255): 134.2670273047587, (266, 264): 166.9949493661168}, (181, 153): {(270, 236): 155.39696961966993, (251, 229): 139.29646455628162, (264, 231): 151.46803743153546, (250, 229): 138.29646455628162, (247, 230): 134.88225099390849, (267, 264): 160.68124086713206, (242, 233): 128.63961030678917, (243, 264): 136.6812408671318, (237, 255): 126.95331880577389, (266, 264): 159.68124086713203}, (184, 179): {(270, 236): 128.1543289325506, (251, 229): 112.05382386916226, (264, 231): 124.22539674441607, (250, 229): 111.05382386916226, (247, 230): 107.63961030678917, (267, 264): 133.43860018001251, (242, 233): 101.39696961966989, (243, 264): 109.43860018001246, (237, 255): 99.71067811865463, (266, 264): 132.43860018001251}, (187, 162): {(270, 236): 127.1248916810277, (251, 229): 112.25483399593895, (264, 231): 119.63961030678915, (250, 229): 111.84062043356586, (247, 230): 111.59797974644658, (267, 264): 153.88225099390868, (242, 233): 112.52691193458112, (243, 264): 143.94112549695438, (237, 255): 132.45584412271572, (266, 264): 153.4680374315356}, (270, 236): {(65, 136): 287.6101730552664, (73, 99): 288.3086578651016, (84, 133): 267.36753236814707, (53, 119): 300.0243866176395, (69, 137): 283.7817459305204, (72, 98): 289.72287142747473, (59, 132): 291.95331880577396, (55, 108): 302.58073580374366, (68, 99): 293.3086578651016, (85, 104): 274.2375900532361}, (251, 229): {(65, 136): 272.7401153701776, (73, 99): 273.4386001800128, (84, 133): 252.4974746830583, (53, 119): 285.1543289325507, (69, 137): 269.1543289325507, (72, 98): 274.85281374238593, (59, 132): 277.08326112068517, (55, 108): 287.71067811865487, (68, 99): 278.4386001800128, (85, 104): 259.3675323681473}, (264, 231): {(65, 136): 280.1248916810278, (73, 99): 280.82337649086304, (84, 133): 259.8822509939085, (53, 119): 292.5391052434009, (69, 137): 276.5391052434009, (72, 98): 282.23759005323615, (59, 132): 284.4680374315354, (55, 108): 295.0954544295051, (68, 99): 285.82337649086304, (85, 104): 266.7523086789975}, (250, 229): {(65, 136): 272.32590180780454, (73, 99): 273.0243866176398, (84, 133): 252.08326112068522, (53, 119): 284.74011537017765, (69, 137): 268.74011537017765, (72, 98): 274.4386001800129, (59, 132): 276.6690475583121, (55, 108): 287.2964645562818, (68, 99): 278.0243866176398, (85, 104): 258.95331880577424}, (247, 230): {(65, 136): 272.0832611206853, (73, 99): 272.78174593052046, (84, 133): 251.84062043356593, (53, 119): 284.4974746830584, (69, 137): 268.49747468305833, (72, 98): 274.19595949289356, (59, 132): 276.42640687119285, (55, 108): 287.05382386916256, (68, 99): 277.78174593052046, (85, 104): 258.7106781186549}, (267, 264): {(65, 136): 293.4802307403554, (73, 99): 297.4924240491755, (84, 133): 275.72287142747473, (53, 119): 309.20815280171337, (69, 137): 289.0660171779823, (72, 98): 298.9066376115486, (59, 132): 301.13708498984784, (55, 108): 311.76450198781754, (68, 99): 302.4924240491755, (85, 104): 283.42135623730996}, (242, 233): {(65, 136): 273.0121933088198, (73, 99): 273.71067811865504, (84, 133): 252.76955262170048, (53, 119): 285.4264068711929, (69, 137): 269.4264068711929, (72, 98): 275.12489168102815, (59, 132): 277.3553390593274, (55, 108): 287.9827560572971, (68, 99): 278.71067811865504, (85, 104): 259.6396103067895}, (243, 264): {(65, 136): 269.48023074035524, (73, 99): 273.4924240491754, (84, 133): 251.72287142747456, (53, 119): 285.20815280171325, (69, 137): 265.06601717798213, (72, 98): 274.9066376115485, (59, 132): 277.1370849898477, (55, 108): 287.7645019878174, (68, 99): 278.4924240491754, (85, 104): 259.42135623730985}, (237, 255): {(65, 136): 259.7523086789974, (73, 99): 263.76450198781754, (84, 133): 241.9949493661167, (53, 119): 275.4802307403554, (69, 137): 255.3380951166243, (72, 98): 265.17871555019065, (59, 132): 267.4091629284899, (55, 108): 278.0365799264596, (68, 99): 268.76450198781754, (85, 104): 249.693434175952}, (266, 264): {(65, 136): 292.4802307403554, (73, 99): 296.4924240491755, (84, 133): 274.72287142747473, (53, 119): 308.20815280171337, (69, 137): 288.0660171779823, (72, 98): 297.9066376115486, (59, 132): 300.13708498984784, (55, 108): 310.76450198781754, (68, 99): 301.4924240491755, (85, 104): 282.42135623730996}, (65, 136): {(30, 80): 70.49747468305826, (33, 106): 44.42640687119284, (17, 76): 79.88225099390849, (36, 97): 51.01219330881973, (21, 76): 78.2253967444161, (36, 96): 52.012193308819725, (36, 92): 56.01219330881972, (17, 115): 56.69848480983495, (27, 78): 73.74011537017753, (17, 114): 57.112698372208044}, (73, 99): {(30, 80): 50.870057685088774, (33, 106): 42.899494936611646, (17, 76): 65.52691193458112, (36, 97): 37.828427124746185, (21, 76): 61.52691193458113, (36, 96): 38.24264068711928, (36, 92): 39.89949493661165, (17, 115): 62.627416997969476, (27, 78): 54.69848480983495, (17, 114): 62.213203435596384}, (84, 133): {(30, 80): 75.95331880577396, (33, 106): 62.18376618407351, (17, 76): 90.61017305526632, (36, 97): 62.91168824543137, (21, 76): 86.61017305526633, (36, 96): 63.32590180780446, (36, 92): 64.98275605729685, (17, 115): 74.45584412271566, (27, 78): 79.78174593052015, (17, 114): 74.87005768508875}, (53, 119): {(30, 80): 48.526911934581165, (33, 106): 25.384776310850246, (17, 76): 57.911688245431385, (36, 97): 29.041630560342625, (21, 76): 56.254833995939, (36, 96): 30.041630560342625, (36, 92): 34.041630560342625, (17, 115): 37.65685424949237, (27, 78): 51.76955262170044, (17, 114): 38.07106781186546}, (69, 137): {(30, 80): 73.15432893255064, (33, 106): 48.840620433565924, (17, 76): 82.53910524340085, (36, 97): 53.66904755831211, (21, 76): 80.88225099390849, (36, 96): 54.6690475583121, (36, 92): 58.669047558312094, (17, 115): 61.11269837220804, (27, 78): 76.39696961966992, (17, 114): 61.52691193458113}, (72, 98): {(30, 80): 49.45584412271568, (33, 106): 42.31370849898474, (17, 76): 64.11269837220803, (36, 97): 36.41421356237309, (21, 76): 60.11269837220804, (36, 96): 36.828427124746185, (36, 92): 38.48528137423856, (17, 115): 62.04163056034257, (27, 78): 53.28427124746186, (17, 114): 61.627416997969476}, (59, 132): {(30, 80): 64.0121933088197, (33, 106): 36.76955262170048, (17, 76): 73.39696961966992, (36, 97): 44.52691193458117, (21, 76): 71.74011537017755, (36, 96): 45.52691193458117, (36, 92): 49.52691193458116, (17, 115): 49.04163056034258, (27, 78): 67.25483399593898, (17, 114): 49.45584412271568}, (55, 108): {(30, 80): 38.35533905932738, (33, 106): 22.82842712474619, (17, 76): 51.254833995939016, (36, 97): 23.556349186104054, (21, 76): 47.25483399593902, (36, 96): 23.97056274847715, (36, 92): 25.62741699796953, (17, 115): 40.899494936611646, (27, 78): 41.597979746446654, (17, 114): 40.485281374238554}, (68, 99): {(30, 80): 45.87005768508879, (33, 106): 37.89949493661165, (17, 76): 60.52691193458113, (36, 97): 32.82842712474619, (21, 76): 56.52691193458114, (36, 96): 33.242640687119284, (36, 92): 34.89949493661166, (17, 115): 57.627416997969476, (27, 78): 49.698484809834966, (17, 114): 57.213203435596384}, (85, 104): {(30, 80): 64.94112549695421, (33, 106): 52.828427124746185, (17, 76): 79.59797974644658, (36, 97): 51.899494936611646, (21, 76): 75.59797974644658, (36, 96): 52.31370849898474, (36, 92): 53.97056274847711, (17, 115): 72.55634918610401, (27, 78): 68.7695526217004, (17, 114): 72.14213562373092}}
        total_distance, selected_samples = self.select_neighbours(sequence, neighbour_distances, neighbors)
        path = []
        for a in range(0,n):
            b = (a+1) % n
            actual_path = planner.plan(gridmap, selected_samples[a], selected_samples[b], neigh = 'N8')
            path = path + actual_path

        return path

    def noon_bean_transform(self, neighbors, pairwise_distances):
        max_dist = -1
        for i in range(len(neighbors)):
            for k1,v in pairwise_distances.items():
                k2 = max(v)
                if v[k2] > max_dist:
                    max_dist = v[k2]
                    max_key = (k1, k2)
        flat_neighbours = [n for neighbourhood in neighbors for n in neighbourhood]
        M = len(flat_neighbours)*max_dist

        d = defaultdict(dict)
        N = len(neighbors)
        for i in range(N):
            l = len(neighbors[i])
            for a in range(l):
                for j in range(N):
                    k = len(neighbors[j])
                    for b in range(k):
                        g1 = neighbors[i][a]
                        g2 = neighbors[j][b]
                        new_g2 = neighbors[j][(b+1) % k]
                        d[g1][new_g2] = M + pairwise_distances[g1][g2]

        for i in range(len(neighbors)):
            l = len(neighbors[i])
            for a in range(l):
                b = (a+1) % l
                d[neighbors[i][a]][neighbors[i][b]] = 0

        return d

    def adjust_sequence_NB(self, sequence, neighbors, flat_neighbours):
        city_size = len(neighbors[0])
        probe = sequence[0:city_size]
        for city in neighbors:
            if flat_neighbours[probe[0]] in city: break
        count = sum([1 for x in probe if flat_neighbours[x] in city])
        return sequence[-(city_size-count):] + sequence[:-(city_size-count)]

    def select_neighbours_NB(self, sequence, neighbour_distances, neighbors):
        city_size = len(neighbors[0])
        flat_neighbours = [x for neighbourhood in neighbors for x in neighbourhood]
        array = [[flat_neighbours[j] for j in [sequence[i], sequence[i + city_size - 1]]] for i in range(len(sequence)) if i % city_size == 0]
        # array = [[flat_neighbours[j] for j in sequence[i:(i+city_size)]] for i in range(len(sequence)) if i % city_size == 0] #all

        n = len(sequence)
        best = np.inf
        sample = []
        for permutation in itertools.product(*array):
            total = 0
            m = len(permutation)
            for a in range(0,m):
                b = (a+1) % m
                total += neighbour_distances[permutation[a]][permutation[b]]
                if total > best: break
            if total <= best:
                best = total
                sample = permutation

        return best, sample

    def plan_tour_obstacles_neighborhood_NB(self, gridmap, goals, neighbors, start_idx = None, end_idx = None):	
        planner = gplanner.GridPlanner()
        flat_neighbours = [n for neighbourhood in neighbors for n in neighbourhood]
        neighbour_distances, all_pairs_paths = self.get_all_neighbourhood_distances(gridmap, flat_neighbours, planner)
        neighbour_distances_NB = self.noon_bean_transform(neighbors, neighbour_distances)
        
        n = len(flat_neighbours)
        self.distances = np.zeros((n,n))    
        self.paths = {}
        for a in range(n): 
            for b in range(n):
                if a != b:
                    g1 = flat_neighbours[a]
                    g2 = flat_neighbours[b]
                    self.distances[a][b] = neighbour_distances_NB[g1][g2]
                    self.paths[(a,b)] = all_pairs_paths[g1][g2]

        sequence = self.compute_TSP_sequence(start_idx, end_idx)
        sequence = self.adjust_sequence_NB(sequence, neighbors, flat_neighbours)
        city_size = len(neighbors[0])
        # sample = [flat_neighbours[sequence[i]] for i in range(city_size-1, len(sequence), city_size)]
        best, sample = self.select_neighbours_NB(sequence, neighbour_distances, neighbors)
        # print("BEST: " + str(best))
        n = len(sample)
        # n = len(sequence) #test whole
        path = []
        for a in range(0, n):
            b = (a+1) % n
            actual_path = planner.plan(gridmap, sample[a], sample[b], neigh = 'N8')
            # actual_path = planner.plan(gridmap, flat_neighbours[sequence[a]], flat_neighbours[sequence[b]], neigh = 'N8') #test whole
            path = path + actual_path
        return path
     
    
