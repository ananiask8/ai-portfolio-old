# -*- coding: utf-8 -*-
"""

Dubins TSP with Neighborhoods (DTSPN)

@author: P.Vana & P.Cizek
"""
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import random

from invoke_LKH import *

#import dubins
from DubinsManeuver import DubinsManeuver as dubins

def dist_euclidean_squared(coord1, coord2):
    (x1, y1) = coord1
    (x2, y2) = coord2
    return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)

def dist_euclidean(coord1, coord2):
    return math.sqrt(dist_euclidean_squared(coord1, coord2))

class DTSPNSolver:
    def __init__(self):
        pass

    # compute the shortest sequence based on the distance matrix (self.distances)
    def compute_TSP_sequence(self, start_idx = None, end_idx = None):
        n = len(self.distances)
        
        if start_idx != None and end_idx != None:
            # TODO - insert your code here
            print ("TODO")
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
        
        if start_idx != None and end_idx != None:
            path = []
            for a in range(start_idx,start_idx+n-1):
                b = (a+1) % n
                a_idx = sequence[a]
                b_idx = sequence[b]
                actual_path = self.paths[(a_idx,b_idx)]
                path = path + actual_path
        else:
            path = []
            for a in range(0,n):
                b = (a+1) % n
                a_idx = sequence[a]
                b_idx = sequence[b]
                actual_path = self.paths[(a_idx,b_idx)]
                path = path + actual_path
        
        return path
        
    
    def plan_tour_etsp(self, goals):
        n = len(goals)
        self.distances = np.zeros((n,n))	
        self.paths = {}

        # find path between each pair of goals (a,b)
        for a in range(0,n): 
            for b in range(0,n):
                g1 = goals[a]
                g2 = goals[b]
                if a != b:
                    # store distance
                    self.distances[a][b] = dist_euclidean(g1, g2) 
                    # store path
                    self.paths[(a,b)] = [g1, g2]

        return self.compute_TSP_tour()

    # for a particular city
    def get_waypoint_samples(self, center, radius, current_opt=None, resolution=None, samples=8):
        waypoints = []
        if current_opt is None:
            sampled_r = np.random.uniform(0, 2*np.pi, samples)
        else:
            dif = (current_opt[0] - center[0], current_opt[1] - center[1])
            theta = np.arctan2(dif[1], dif[0])
            theta = theta if theta > 0 else theta + 2*np.pi
            sampled_r = np.random.uniform(theta - np.pi/resolution, theta + np.pi/resolution, samples)
        waypoints += [np.array(center) + radius*np.array([np.cos(r), np.sin(r)]) for r in sampled_r]
        return waypoints

    # for a particular city 1
    def get_configurations_with_headings(self, sequence, waypoints, resolution=None, samples=8):
        n = len(waypoints)
        sampled_headings = np.random.uniform(0, 2*np.pi, samples)
        configurations = [[] for i in range(n)]
        for h in sampled_headings:
            for i in range(n):
                city_configurations = []
                for origin in waypoints[sequence[i]]:
                    for next_waypoint in waypoints[sequence[(i+1)%n]]:
                        # print(i, waypoint, next_waypoint)                            
                        dif = (next_waypoint[0] - origin[0], next_waypoint[1] - origin[1])
                        theta = np.arctan2(dif[1], dif[0])
                        theta = theta if theta > 0 else theta + 2*np.pi
                        if theta + np.pi/resolution > h and theta - np.pi/resolution < h:
                            city_configurations.append((origin[0], origin[1], h))
                            break # if not would add the same origin,h
                # configurations[sequence[i]] += city_configurations if len(city_configurations) > 0 else [(origin[0], origin[1], theta)]
                configurations[sequence[i]] += city_configurations
        return configurations

    def compute_distances_vertex_in_city_to_vertex_in_another_city(self, sequence, configurations, turning_radius):
        n = len(configurations)
        distances = defaultdict(dict)
        for i in range(n):
            for j in range(n):
                ith_city_sequence_idx, jth_city_sequence_idx = sequence.index(i), sequence.index(j)
                if i == j or ( \
                    ( \
                        abs(ith_city_sequence_idx - jth_city_sequence_idx) > 1 or
                        ith_city_sequence_idx > jth_city_sequence_idx
                    ) and \
                    not (ith_city_sequence_idx == n - 1 and jth_city_sequence_idx == 0)): continue
                for a in configurations[i]:
                    for b in configurations[j]:
                        distances[a][b] = dubins.shortest_path(a, b, turning_radius).get_length()
        return distances

    def create_configuration_graph(self, sequence, centers, sensing_radius, turning_radius, resolution=1, current_opt=None):
        n = len(centers)
        waypoints = []
        graph = []
        for i in range(n):
            opt = None if current_opt is None else current_opt[sequence.index(i)]
            c = centers[i]
            waypoints.append(self.get_waypoint_samples(c, sensing_radius, opt, resolution))

        graph = self.get_configurations_with_headings(sequence, waypoints, resolution)
        if current_opt is not None:
            for i in range(n):
                #reinsert previous optimal solution to start optimizing only from it
                graph[sequence[i]].insert(0, current_opt[i])

        w = self.compute_distances_vertex_in_city_to_vertex_in_another_city(sequence, graph, turning_radius)
        return graph, w

    def select_neighbours(self, sequence, w, graph, limit=100):
        n = len(sequence)
        sample = [0]*n
        # always best index is 0 since in waypoints I am adding the previous optimum at the beginning
        best = sum([w[graph[sequence[i]][0]][graph[sequence[(i+1)%n]][0]] for i in range(n)])
        improving = True
        count = 0
        indeces = list(range(n))
        np.random.shuffle(indeces)
        while improving:
            if count > limit: break
            count += 1
            improving = False
            for j in indeces:
                i = (j-1) % n
                k = (j+1) % n
                prev_idx, current_idx, next_idx = sequence[i], sequence[j], sequence[k]
                prev_r, current_r, next_r = graph[prev_idx], graph[current_idx], graph[next_idx]
                prev_v_idx, current_v_idx, next_v_idx = sample[i], sample[j], sample[k]
                prev_v, current_v, next_v = prev_r[prev_v_idx], current_r[current_v_idx], next_r[next_v_idx]
                current_add_component = w[prev_v][current_v] + w[current_v][next_v]

                m = len(current_r)
                for new_v_idx in range(m):
                    new_v = current_r[new_v_idx]
                    new_add_component = w[prev_v][new_v] + w[new_v][next_v]
                    if best - current_add_component + new_add_component < best and random.random() > 0.5:
                        improving = True
                        best += new_add_component - current_add_component
                        sample[j] = new_v_idx
                        break # necessary, or save new add component to current add component
        return best, [graph[sequence[i]][sample[i]] for i in range(n)]

    def plan_tour_decoupled(self, goals, sensing_radius, turning_radius):
        n = len(goals)
        self.distances = np.zeros((n,n))	
        self.paths = {}

        # find path between each pair of goals (a,b)
        for a in range(0,n): 
            for b in range(0,n):
                g1 = goals[a]
                g2 = goals[b]
                if a != b:
                    # store distance
                    self.distances[a][b] = dist_euclidean(g1, g2) 
        
        sequence = self.compute_TSP_sequence()
        sample, sample_idx = None, None
        for resolution in [1.]*30 + [2.]*30 + [4.]*20 + [8.]*20 + [16.]*10 + [32.]*10 + [64.]*10:
            graph, w = self.create_configuration_graph(sequence, goals, sensing_radius, turning_radius, resolution, sample)
            best, sample = self.select_neighbours(sequence, w, graph)

        print(best)
        path = []
        for a in range(0, n):
            b = (a+1) % n
            start = sample[a]
            end = sample[b]
            step_size = 0.01 * turning_radius
            # print(goals[sequence[a]], start)
            dubins_path = dubins.shortest_path(start, end, turning_radius)
            configurations, _ = dubins_path.sample_many(step_size)
            path = path + configurations

        return path
    
    
     
     
    
