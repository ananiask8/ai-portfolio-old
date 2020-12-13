#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import itertools
from scipy.optimize import linprog
import multiprocessing as mp

import GridMap as gmap
import GridPlanner as gplanner
from MCTS import MCTS, InstanceHelper
from pathlib import Path

import time
import random
import os
import pickle

PURSUER = 1
EVADER = 2

GREEDY = "GREEDY"
MONTE_CARLO = "MONTE_CARLO"
VALUE_ITERATION = "VALUE_ITERATION"

class Player:
    def __init__(self, robots, role, policy=GREEDY, color='r', timeout=5.0, game=None):
        """
        Parameters
        ----------
        robots: list((in,int))
            coordinates of individual player's robots
        role: int
            player's role in the game
        """
        #list of the player's robots
        self.robots = robots[:]
        #next position of the player's robots
        self.next_robots = robots[:]

        if role == "EVADER":
            self.role = EVADER
        elif role == "PURSUER":
            self.role = PURSUER
        else:
            raise ValueError('Unknown player role')

        #selection of the policy
        if policy == GREEDY:
            self.policy = self.greedy_policy
        elif policy == MONTE_CARLO:
            self.policy = self.monte_carlo_policy
        elif policy == VALUE_ITERATION:
            self.policy = self.value_iteration_policy
        else:
            raise ValueError('Unknown policy')

        self.color = color #color for plotting purposes
        self.timeout = timeout #planning timeout fo MCST
        self.game_name = game #game name for loading value iteration policies
        self.distances = None

        #values for the value iteration policy
        self.values = None
        self.mcts = {PURSUER: {}, EVADER: {}}

        #instantiation of helper planner
        self.Planner = gplanner.GridPlanner()
    
    #####################################################
    # Game interface functions
    #####################################################
    def add_robot(self, pos):
        """
        method to add a robot to the player
        
        Parameters
        ----------
        pos: (int,int)
            position of the robot
        """
        self.robots.append(pos)
        self.next_robots.append(pos)
    
    def del_robot(self, pos):
        """
        method to remove the player's robot 

        Parameters
        ----------
        pos: (int,int)
            position of the robot to be removed
        """
        idx = self.robots.index(pos)
        self.robots.pop(idx)
        self.next_robots.pop(idx)
    
    def calculate_step(self, gridmap, evaders, pursuers):
        """
        method to calculate the player's next step using selected policy
        
        Parameters
        ----------
        gridmap: GridMap
            map of the environment
        evaders: list((int,int))
            list of coordinates of evaders in the game (except the player's robots, if he is evader)
        pursuers: list((int,int))
            list of coordinates of pursuers in the game (except the player's robots, if he is pursuer)
        """
        self.policy(gridmap, evaders, pursuers)
    
    def take_step(self):
        """
        method to perform the step 
        """
        self.robots = self.next_robots[:]

    #####################################################
    #####################################################
    # FLOYD WARSHALL
    #####################################################
    #####################################################
    def get_array_of_coordinates(self, n, m):
        coords = []
        for i in range(n):
            for j in range(m):
                coords.append((i, j))
        return coords

    def prepare_weight_matrix(self, gridmap):
        coords = self.get_array_of_coordinates(gridmap.width, gridmap.height)
        w = defaultdict(dict)
        for coord1 in coords:
            for coord2 in coords:
                if gridmap.passable(coord1) and gridmap.passable(coord2):
                    w[coord1][coord2] = self.dist(gridmap, coord1, coord2)
                else:
                    w[coord1][coord2] = np.inf
        return w

    def floyd_warshall(self, gridmap, w):
        # coords = list(filter(lambda coord: gridmap.passable(coord), self.get_array_of_coordinates(gridmap.width, gridmap.height)))
        coords = self.get_array_of_coordinates(gridmap.width, gridmap.height)
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

    def select_closest_to_closest_evader(self, neighbors, evaders, distances):
        best = np.inf
        best_neighbor = None
        for n in neighbors:
            for e in evaders:
                if distances[n][e] < best:
                    best = distances[n][e]
                    best_neighbor = n

        return best_neighbor

    def select_most_distant_to_pursuers(self, neighbors, pursuers, distances):
        best = -1
        best_neighbor = None
        for n in neighbors:
            d = np.inf
            for p in pursuers:
                d = min(d, distances[n][p])
            if d > best:
                best = d
                best_neighbor = n

        return best_neighbor
        
    #####################################################
    #####################################################
    # GREEDY POLICY
    #####################################################
    #####################################################
    def greedy_policy(self, gridmap, evaders, pursuers, epsilon=1):
        """
        Method to calculate the greedy policy action
        
        Parameters
        ----------
        gridmap: GridMap
            Map of the environment
        evaders: list((int,int))
            list of coordinates of evaders in the game (except the player's robots, if he is evader)
        pursuers: list((int,int))
            list of coordinates of pursuers in the game (except the player's robots, if he is pursuer)
        epsilon: float (optional)
            optional epsilon-greedy parameter
        """
        self.next_robots = self.robots[:]
        if self.distances is None:
            w = self.prepare_weight_matrix(gridmap)
            self.distances, _ = self.floyd_warshall(gridmap, w)

        #for each of player's robots plan their actions
        for idx in range(0, len(self.robots)):
            robot = self.robots[idx]
            neighbors = gridmap.neighbors4(robot)
            if random.random() < epsilon:
                #select the next action based on own role
                if self.role == PURSUER:
                    self.next_robots[idx] = self.select_closest_to_closest_evader(neighbors, evaders, self.distances)
                elif self.role == EVADER:
                    self.next_robots[idx] = self.select_most_distant_to_pursuers(neighbors, pursuers, self.distances)
            else:
                ##################################################
                # RANDOM Policy
                ##################################################
                #introducing randomness in neighbor selection
                random.shuffle(neighbors)
                #select random goal
                pos_selected = neighbors[0]
                self.next_robots[idx] = pos_selected
                ##################################################

    #####################################################
    #####################################################
    # MONTE CARLO TREE SEARCH POLICY
    #####################################################
    #####################################################
    def monte_carlo_policy(self, gridmap, evaders, pursuers):
        """
        Method to calculate the monte carlo tree search policy action
        
        Parameters
        ----------
        gridmap: GridMap
            Map of the environment
        evaders: list((int,int))
            list of coordinates of evaders in the game (except the player's robots, if he is evader)
        pursuers: list((int,int))
            list of coordinates of pursuers in the game (except the player's robots, if he is pursuer)
        """
        self.next_robots = self.robots[:]
        
        if self.distances is None:
            w = self.prepare_weight_matrix(gridmap)
            self.distances, _ = self.floyd_warshall(gridmap, w)

        #for each of player's robots plan their actions
        print("PURSUER" if self.role == PURSUER else "EVADER")
        for idx in range(0, len(self.robots)):
            robot = self.robots[idx]
            if self.role == PURSUER:
                Ne, Np = len(evaders), len(self.robots)
                s = tuple(evaders) + tuple(self.robots)
                opponent = evaders
                i = Ne + idx
            else:
                Ne, Np = len(self.robots), len(pursuers)
                s = tuple(self.robots) + tuple(pursuers)
                opponent = pursuers
                i = idx
            if s not in self.mcts[self.role]:
                ih = InstanceHelper(self.robots, opponent, self.role, self.distances, gridmap, self.select_closest_to_closest_evader, self.select_most_distant_to_pursuers)
                self.mcts[self.role][s] = MCTS(s, ih=ih, limit=5)
                self.next_robots[idx] = self.mcts[self.role][s].get_best_next_state()[i]
            else:
                self.next_robots[idx] = self.mcts[self.role][s].get_best_next_state()[i]


    #####################################################
    #####################################################
    # VALUE ITERATION POLICY
    #####################################################
    #####################################################
    def get_required_action(self, s, s_goal):
        a = np.array([np.array(s_goal[i]) - np.array(s[i]) for i in range(len(s))])
        return None if np.any(np.sum(a, axis=1) > 1) or np.any(np.sum(-a, axis=1) < -1) else tuple([tuple(ai) for ai in a])

    def generate_transition_states(self, gridmap, s, A):
        goals = []
        for a_all in itertools.product(*[A]*len(s)):
            g = [tuple([s[i][j] + a_all[i][j] for j in range(len(s[i]))]) for i in range(len(s))]
            if not np.all([gridmap.passable(gi) and gridmap.in_bounds(gi) for gi in g]): continue
            goals.append((a_all, tuple([tuple(gi) for gi in g])))
        return goals

    def has_capture(self, s):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        p = s[slicing_idx:] 
        e = s[:slicing_idx]
        return True if np.any([e1 in p for e1 in e]) else False

    def any_crossing(self, si, sj):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        pi, ei = si[slicing_idx:], si[:slicing_idx]
        pj, ej = sj[slicing_idx:], sj[:slicing_idx]
        for p_idx in range(len(pi)):
            for e_idx in range(len(ei)):
                if pj[p_idx] == ei[e_idx] and pi[p_idx] == ej[e_idx]: return True
        return False

    def get_reward(self, s, a, s_goal):
        return 1 if self.has_capture(s_goal) or self.any_crossing(s, s_goal) else 0

    def generate_states(self, gridmap, evaders, pursuers):
        grid_positions = [(i,j) for j in range(gridmap.height) for i in range(gridmap.width) if gridmap.passable((i,j))]
        
        ###############
        t1 = time.time()
        S = itertools.product(*[grid_positions]*(len(self.robots) + len(evaders) + len(pursuers)))
        t2 = time.time()
        print(t2-t1)
        ###############
        
        A = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        P, R = defaultdict(dict), defaultdict(dict)
        values = {}

        ###############
        t1 = time.time()
        for s in S:
            # for j, s_goal in enumerate(S):
            #     a = self.get_required_action(s, s_goal)
            #     if i == j or a is None: continue

            #     P[s][a] = [s_goal] if a not in P[s] else P[s][a] + [s_goal]
            #     R[s][a] = self.get_reward(s, a, s_goal)
            values[s] = 0
            for a, s_goal in self.generate_transition_states(gridmap, s, A):
                P[s][a] = [s_goal]
                R[s][a] = self.get_reward(s, a, s_goal)
        t2 = time.time()
        print(t2-t1)
        print("CHECKPOINT")
        ###############
        return P, R, values

    def initialize_values(self, S):
        return {s: 0 for s in S}

    def maxmin(self, Q):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        Ae = list(set([k[:slicing_idx] for k in Q.keys()])) # list due to consistency of LP variables
        Ap = list(set([k[slicing_idx:] for k in Q.keys()]))

        c = [-1] + [0]*len(Ap)
        bounds = []
        ub = np.inf
        Mu, Me = [], []
        Me.append([0] + [1]*len(Ap))
        for j in Ae:
            Mu.append([1] + [(-Q[j+i] if (j+i in Q) else 0) for i in Ap])
        be = [1]
        bu = [0]*len(Ae)
        bounds = tuple([(None, None)] + [(0, None) for _ in Ap])
        res = linprog(c, A_ub=Mu, b_ub=bu, A_eq=Me, b_eq=be, bounds=bounds)
        # print(res.x[0])
        return res.x[0] if res.success else None

    def argmax(self, Q):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        Ae = list(set([k[:slicing_idx] for k in Q.keys()])) # list due to consistency of LP variables
        Ap = list(set([k[slicing_idx:] for k in Q.keys()]))
        scores_e, scores_p = {a: 0 for a in Ae}, {a: 0 for a in Ap}
        for i in Ae:
            for j in Ap:
                scores_e[i] += Q[i+j]
                scores_p[j] += Q[i+j]
        # print(Q)
        # print(scores_e, scores_p)
        return min(scores_e, key=lambda a: scores_e[a]), max(scores_p, key=lambda a: scores_p[a])

    def parallel_hotness(self, s):
        Q = {a: self.R[s][a] + self.gamma*np.sum([self.values[s_goal] for s_goal in self.P[s][a]]) for a in self.P[s].keys()}
        return {s: self.maxmin(Q)}

    def dirname(self, path):
        return '/'.join(str(path).split('/')[:-1])

    def value_iteration_policy(self, gridmap, evaders, pursuers):
        """
        Method to calculate the value-iteration policy action
        
        Parameters
        ----------
        gridmap: GridMap
            Map of the environment
        evaders: list((int,int))
            list of coordinates of evaders in the game (except the player's robots, if he is evader)
        pursuers: list((int,int))
            list of coordinates of pursuers in the game (except the player's robots, if he is pursuer)
        """
        self.next_robots = self.robots[:]

        #if there are not precalculated values for policy
        if not self.values: 
            policy_file = Path("policies/" + self.game_name + ".policy")
            ################################################### 
            #if there is policy file, load it...
            ################################################### 
            if policy_file.is_file():
                #load the strategy file
                self.values = pickle.load(open(str(policy_file), 'rb'))[0]

            ################################################### 
            #...else calculate the policy
            ################################################### 
            else:
                self.P, self.R, self.values = self.generate_states(gridmap, evaders, pursuers)
                # print("INITIALIZED")
                v = {}
                # Q = defaultdict(dict)
                self.gamma = 0.9
                # pool = mp.Pool(processes=mp.cpu_count())
                while True:
                    # Q = defaultdict(dict)
                    ###############
                    t1 = time.time()
                    for s in self.P.keys():
                        Q = {a: self.R[s][a] + self.gamma*np.sum([self.values[s_goal] for s_goal in self.P[s][a]]) for a in self.P[s].keys()}
                        v[s] = self.maxmin(Q)
                    # v = {k: val for vs in pool.map(self.parallel_hotness, [s for s in self.P.keys()]) for k, val in vs.items()}
                    # print(np.array([abs(self.values[s] - v[s]) for s in v.keys()]))
                    t2 = time.time()
                    print(t2 - t1)
                    ###############

                    if np.all(np.array([abs(self.values[s] - v[s]) for s in v.keys()]) < 5e-1): break
                    self.values = v.copy()
                self.values = v

                #save the policy
                os.makedirs(self.dirname(policy_file), exist_ok=True)
                pickle.dump([self.values], open(str(policy_file),'wb'))
        # print(self.values)
        for idx in range(0, len(self.robots)):
            robot = self.robots[idx]
            if self.role == PURSUER:
                Ne, Np = len(evaders), len(self.robots)
                s = tuple(evaders) + tuple(self.robots)
                i = Ne + idx
            else:
                Ne, Np = len(self.robots), len(pursuers)
                s = tuple(self.robots) + tuple(pursuers)
                i = idx
            Q, v = defaultdict(dict), defaultdict(dict)
            A = [(1, 0), (0, -1), (-1, 0), (0, 1)]        
            for a, s_goal in self.generate_transition_states(gridmap, s, A):
                Q[a] = self.get_reward(s, a, s_goal) + self.values[s_goal]
            ae, ap = self.argmax(Q)
            self.next_robots[idx] = tuple(np.array(robot) + np.array(ap[idx] if self.role == PURSUER else ae[idx]))
    
    #####################################################
    # Helper functions
    #####################################################
    def dist(self, gridmap, coord1, coord2):
        #using A* to get shortest path
        pth = self.Planner.plan(gridmap, coord1, coord2, neigh='N4')
        dst = len(pth)
        return dst

