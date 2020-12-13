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
    def __init__(self, step_size):
        self.step_size = step_size

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

    def get_vector_difference_angle(self, origin, goal):
        dif = (goal[0] - origin[0], goal[1] - origin[1])
        theta = np.arctan2(dif[1], dif[0])
        theta = theta if theta > 0 else theta + 2*np.pi
        return theta

    # for a particular city
    def get_waypoint_samples(self, center, radius, current_opt=None, samples=8, resolution=None):
        waypoints = []
        if current_opt is None:
            sampled_r = np.random.uniform(0, 2*np.pi, samples)
        else:
            theta = self.get_vector_difference_angle(center, current_opt)
            sampled_r = np.random.uniform(theta - np.pi/resolution, theta + np.pi/resolution, samples)
        waypoints += [np.array(center) + radius*np.array([np.cos(r), np.sin(r)]) for r in sampled_r]
        return waypoints

    # for a particular city 1
    def get_configurations_with_headings(self, sequence, waypoints, samples=8, resolution=None, skip_even=False):
        n = len(waypoints)
        sampled_headings = np.random.uniform(0, 2*np.pi, samples)
        configurations = [[] for i in range(n)]
        for h in sampled_headings:
            for i in range(n):
                city_configurations = []
                for origin in waypoints[sequence[i]]:
                    for next_waypoint in waypoints[sequence[(i+1)%n]]:
                        theta = self.get_vector_difference_angle(origin[0:2], next_waypoint[0:2])                        
                        if skip_even and i % 2 == 0:
                            city_configurations.append((origin[0], origin[1], theta))
                            break
                        elif theta + np.pi/resolution > h and theta - np.pi/resolution < h:
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

    def create_configuration_graph(self, sequence, centers, sensing_radius, turning_radius, neighborhood_samples, heading_samples, resolution=1, current_opt=None, skip_even=False):
        n = len(centers)
        waypoints = []
        graph = []
        # print(centers)
        for i in range(n):
            s_idx = sequence.index(i)
            opt = None if current_opt is None else current_opt[s_idx]
            waypoints.append(self.get_waypoint_samples(centers[i], sensing_radius, opt, neighborhood_samples, resolution))

        graph = self.get_configurations_with_headings(sequence, waypoints, heading_samples, resolution, skip_even=skip_even)
        for i in range(n):
            #reinsert previous optimal solution to start optimizing only from it
            if current_opt is not None: graph[sequence[i]].insert(0, current_opt[i])

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

    def plan_tour_decoupled(self, goals, sensing_radius, turning_radius, neighborhood_samples=8, heading_samples=8):
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
            graph, w = self.create_configuration_graph(sequence, goals, sensing_radius, turning_radius, neighborhood_samples, heading_samples, resolution, sample)
            best, sample = self.select_neighbours(sequence, w, graph)

        print(best)
        path = []
        for a in range(0, n):
            b = (a+1) % n
            start = sample[a]
            end = sample[b]
            dubins_path = dubins.shortest_path(start, end, turning_radius)
            configurations, _ = dubins_path.sample_many(self.step_size)
            path = path + configurations

        return path

    def create_configuration_graph_dtsp_aa(self, sequence, centers, sensing_radius, turning_radius, neighborhood_samples, heading_samples, resolution=1, current_opt=None, skip_even=False):
        n = len(centers)
        waypoints = []
        graph = []
        # print(centers)
        for i in range(n):
            s_idx = sequence.index(i)
            c = centers[i]
            c_1 = centers[sequence[(s_idx+1)%n]]
            theta = self.get_vector_difference_angle(c, c_1)
            waypoints.append([np.array(list(c) + [theta])])
            opt = None if current_opt is None else current_opt[s_idx]
        
        graph = self.get_configurations_with_headings(sequence, waypoints, heading_samples, resolution, skip_even=skip_even)
        for i in range(n):
            #reinsert previous optimal solution to start optimizing only from it
            if current_opt is not None: graph[sequence[i]].insert(0, current_opt[i])

        w = self.compute_distances_vertex_in_city_to_vertex_in_another_city(sequence, graph, turning_radius)
        return graph, w

    def create_configuration_graph_dtspn_aa(self, sequence, centers, sensing_radius, turning_radius, neighborhood_samples, heading_samples, resolution=1, current_opt=None, skip_even=False):
        # AA for DTSPN
        # It creates configurations in the border of the neighborhood
        # but for even (in the index of the sequence) regions the headings will be oriented
        # towards a specific point in the border of the next city
        n = len(centers)
        waypoints = []
        graph = []
        # print(centers)
        for i in range(n):
            s_idx = sequence.index(i)
            opt = None if current_opt is None else current_opt[s_idx]
            waypoints.append(self.get_waypoint_samples(centers[i], sensing_radius, opt, neighborhood_samples, resolution))
        
        graph = self.get_configurations_with_headings(sequence, waypoints, heading_samples, resolution, skip_even=skip_even)
        for i in range(n):
            #reinsert previous optimal solution to start optimizing only from it
            if current_opt is not None: graph[sequence[i]].insert(0, current_opt[i])

        w = self.compute_distances_vertex_in_city_to_vertex_in_another_city(sequence, graph, turning_radius)
        return graph, w

    def plan_tour_aa(self, goals, sensing_radius, turning_radius, neighborhood_samples, heading_samples):
        # AA for DTSP
        # It takes relaxed solution from ETSP
        # and for even (in the index of the sequence) regions the headings will be oriented
        # towards the center of the upcoming region, whereas in odd regions
        # there will be a range of headings to try
        n = len(goals)
        self.distances = np.zeros((n,n))    
        self.paths = {}

        # find path between each pair of goals (a,b)
        for a in range(0,n): 
            for b in range(0,n):
                if a != b: self.distances[a][b] = dist_euclidean(goals[a], goals[b]) 
        
        sequence = self.compute_TSP_sequence()
        sample, sample_idx = None, None
        for resolution in [1.]*10 + [2.]*10 + [4.]*10 + [8.]*10:
            graph, w = self.create_configuration_graph_dtspn_aa(sequence, goals, sensing_radius, turning_radius, neighborhood_samples, heading_samples, resolution, sample, skip_even=True)
            best, sample = self.select_neighbours(sequence, w, graph)
            # print(best)

        path = []
        for a in range(0, n):
            b = (a+1) % n
            dubins_path = dubins.shortest_path(sample[a], sample[b], turning_radius)
            configurations, _ = dubins_path.sample_many(self.step_size)
            path = path + configurations

        return path

    def self_normalize_angle(self, theta):
        while theta < 0:
            theta += 2.*np.pi
        while theta > 2.*np.pi:
            theta -= 2.*np.pi
        return theta

    def create_intervals(self, sequence, goals, eps, samples=8):
        n = len(sequence)
        intervals = [[] for i in range(n)]
        for i in range(n):
            a = sequence[i]
            b = sequence[(i+1)%n]
            theta = self.get_vector_difference_angle(goals[a], goals[b])
            intervals += np.random.uniform(self.normalize_angle(theta - eps), self.normalize_angle(theta + eps, samples))
        return intervals

    def refine_dtp(self, sequence, eps, intervals):
        # TODO
        n = len(sequence)
        # determining the distances from the new intervals
        for i in range(n):
            a = sequence[i]
            b = sequence[(i+1)%n]
            for x in intervals[i]:
                for y in intervals[(i+1)%n]:
                    if len(self.distances[x][y]) == 0:
                        self.distances[x][y] = dubins.shortest_path(x, y, self.turning_radius)

        # selecting best distances from intervals
        new_intervals = [[] for i in range(n)]
        new_costs = [0 for i in range(n)]
        for i in range(n):
            best = np.inf
            for x in intervals[i]:
                for y in intervals[(i+1)%n]:
                    if self.distances[x][y] < best:
                        best = self.distances[x][y]
                        new_intervals[i] = x
                        new_costs[i] = best

        return new_intervals, new_costs

    def solve_dtp(self, sequence, intervals):
        pass

    def sampling_dtp(self, sensing_radius, turning_radius, neighborhood_samples=8, heading_samples=8):
        # TODO
        eps_req = 2*np.pi/128.
        quality_req = 1.01

        n = len(goals)
        self.distances = np.zeros((n,n))
        self.paths = {}
        self.sensing_radius = sensing_radius
        self.turning_radius = turning_radius

        # find path between each pair of goals (a,b)
        for a in range(0,n): 
            for b in range(0,n):
                if a != b: self.distances[a][b] = dist_euclidean(goals[a], goals[b]) 
        sequence = self.compute_TSP_sequence()

        eps = 2*np.pi
        intervals = create_intervals(sequence, eps)
        lb = 0
        ub = np.inf
        while eps > eps_req and ub/lb > quality_req:
            eps = eps/2.
            new_intervals, lb = self.refine_dtp(sequence, eps, intervals)
            tour, lb = self.solve_dtp(sequence, new_intervals)

        return tour

    
     
     
    
