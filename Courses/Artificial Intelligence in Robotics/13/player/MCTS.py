#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import itertools
import random
from collections import defaultdict
import pprint as pp

PURSUER = 1
EVADER = 2

class InstanceHelper:
    def __init__(self, robots, opponent, role, distances, gridmap, select_closest_to_closest_evader, select_most_distant_to_pursuers):
        self.robots = robots
        self.opponent = opponent
        self.role = role
        self.distances = distances
        self.gridmap = gridmap
        self.select_closest_to_closest_evader = select_closest_to_closest_evader
        self.select_most_distant_to_pursuers = select_most_distant_to_pursuers

    def get_maximizer_indeces(self):
        n = len(self.robots)
        total = n + len(self.opponent)
        slicing_idx = -n if self.role == PURSUER else n
        return list(range(total)[slicing_idx:total])

    def get_minimizer_indeces(self):
        n = len(self.robots)
        total = n + len(self.opponent)
        slicing_idx = -n if self.role == PURSUER else n
        return list(range(total)[:slicing_idx])

    def epsilon_greedy(self, s, epsilon=1):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        p = s[slicing_idx:] 
        e = s[:slicing_idx]
        s_goal = []
        for ei in e:
            neighbors = self.gridmap.neighbors4(ei)
            if random.random() < epsilon:
                s_goal.append(self.select_most_distant_to_pursuers(neighbors, p, self.distances))
            else:
                random.shuffle(neighbors)
                s_goal.append(neighbors[0])
        for pi in p:
            neighbors = self.gridmap.neighbors4(pi)
            if random.random() < epsilon:
                s_goal.append(self.select_closest_to_closest_evader(neighbors, e, self.distances))
            else:
                random.shuffle(neighbors)
                s_goal.append(neighbors[0])
        return s_goal

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

    def is_terminal(self, si, sj):
        # print(si, sj, self.any_crossing(si, sj))
        return self.has_capture(sj) or self.any_crossing(si, sj)

    def get_action(self, start, goal):
        a = np.array([np.array(goal[i]) - np.array(start[i]) for i in range(len(start))])
        return None if np.any(np.sum(a, axis=1) > 1) or np.any(np.sum(-a, axis=1) < -1) else tuple([tuple(ai) for ai in a])

    def get_actions(self, start, goal_states):
        return [self.get_action(start, goal) for goal in goal_states]

    def get_next_state(self, s, a):
        return tuple([tuple([s[i][j] + a[i][j] for j in range(len(s[i]))]) for i in range(len(s))])

    def get_reward(self, s, a, s_goal):
        return 1 if self.is_terminal(s, s_goal) else 0

    def neighbors4(self, s):
        return self.gridmap.neighbors4(s)

    def generate_transition_states(self, s, A=[(1, 0), (0, -1), (-1, 0), (0, 1)]):
        goals = []
        for a_all in itertools.product(*[A]*len(s)):
            g = [tuple([s[i][j] + a_all[i][j] for j in range(len(s[i]))]) for i in range(len(s))]
            if not np.all([self.gridmap.passable(gi) and self.gridmap.in_bounds(gi) for gi in g]): continue
            goals.append(tuple(g))
            # goals.append((a_all, tuple([tuple(gi) for gi in g])))
        return goals

    def get_evader_state(self, s):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        p = s[slicing_idx:] 
        e = s[:slicing_idx]
        return e

    def get_pursuer_state(self, s):
        n = len(self.robots)
        slicing_idx = -n if self.role == PURSUER else n
        p = s[slicing_idx:] 
        e = s[:slicing_idx]
        return p

    def get_matrix_game_joint_move(self, maximizer, minimizer):
        # evader minimizes, pursuer maximizes
        return self.get_evader_state(minimizer) + self.get_pursuer_state(maximizer)

class Node:
    def __init__(self, state, action, parent, r, actions, ih, nodes={}):
        self.state = state
        self.action = action
        self.parent = parent
        self.r = r
        self.actions = actions
        self.ih = ih
        self.visited = 0
        self.expanded = False
        self.children = {}
        self.Q = {a: 0. for a in actions}
        self.ni = {a: 0 for a in actions}
        self.nodes = nodes

    def set_action(self, a):
        self.a = a

    def get_children(self):
        return self.children

    def get_child(self, s, a, r, actions):
        if s not in self.children:
            self.children[s] = Node(s, parent=self, action=a, r=r, actions=actions, ih=self.ih, nodes=self.nodes)
        return self.children[s]

    def is_expanded(self):
        return self.expanded

    def expand(self):
        untried_action = None
        for a in self.actions:
            if self.ni[a] == 0:
                untried_action = a
                break

        s = self.ih.get_next_state(self.state, untried_action)
        r = self.ih.get_reward(self.state, untried_action, s)
        actions = self.ih.get_actions(s, self.ih.generate_transition_states(s))
        self.last_node = self.get_child(s, untried_action, self.r + r, actions)
        self.last_action = untried_action
        self.last_node.set_action(untried_action)
        return self.last_node

    def get_best_child_uct_max(self, c, mover_indeces, sign=1):
        # MAX protocol
        v, selected = None, None
        max_value = -np.inf
        uct, all_selected, goals = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for a in self.actions:
            if self.ni[a] == 0: continue
            s = self.ih.get_next_state(self.state, a)
            movers_key = tuple([s[idx] for idx in mover_indeces])
            r = self.ih.get_reward(self.state, a, s)
            if movers_key not in uct: uct[movers_key] = 0
            uct[movers_key] += sign*(self.Q[a]/self.ni[a] + c*np.sqrt((2 * np.log(self.visited)/self.ni[a])))
            all_selected[movers_key] = a
            actions = self.ih.get_actions(s, self.ih.generate_transition_states(s))
            goals[movers_key] = self.get_child(s, a, self.r + r, actions)
        for movers_key in uct.keys():
            if max_value < uct[movers_key]:
                max_value = uct[movers_key]
                selected = all_selected[movers_key]
                v = goals[movers_key]
        # if sign == -1 and c ==0:
        #     pp.pprint(max_value)
        #     pp.pprint(movers_key)
        #     pp.pprint(self.ni)
        #     pp.pprint(self.Q)
        #     pp.pprint(uct)
        #     print(selected, v.state)
        self.last_node = v
        self.last_action = selected
        v.set_action(selected)
        return v

    def get_best_child_uct_mix(self, c, mover_indeces, sign=1):
        # MIX protocol
        v, selected = None, None
        max_value = -np.inf
        uct, all_selected, goals = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for a in self.actions:
            if self.ni[a] == 0: continue
            s = self.ih.get_next_state(self.state, a)
            movers_key = tuple([s[idx] for idx in mover_indeces])
            r = self.ih.get_reward(self.state, a, s)
            if movers_key not in uct: uct[movers_key] = 0
            uct[movers_key] += sign*(self.Q[a]/self.ni[a] + c*np.sqrt((2 * np.log(self.visited) / self.ni[a])))
            all_selected[movers_key] = a
            actions = self.ih.get_actions(s, self.ih.generate_transition_states(s))
            goals[movers_key] = self.get_child(s, a, self.r + r, actions)
        Z = np.sum(list(uct.values()))
        X = list(uct.keys())
        i = np.random.choice(range(len(X)), 1, p=[uct[x]/Z for x in X])[0]
        v, selected = goals[X[i]], all_selected[X[i]]
        self.last_node = v
        self.last_action = selected
        v.set_action(selected)
        return v

    def get_best_child_duct(self, c):
        maximizer = self.get_best_child_uct_max(c, mover_indeces=self.ih.get_maximizer_indeces(), sign=1).state
        minimizer = self.get_best_child_uct_max(c, mover_indeces=self.ih.get_minimizer_indeces(), sign=-1).state # zero sum game
        s = self.ih.get_matrix_game_joint_move(maximizer, minimizer)
        selected = self.ih.get_action(self.state, s)
        r = self.ih.get_reward(self.state, selected, s)
        actions = self.ih.get_actions(s, self.ih.generate_transition_states(s))
        v = self.get_child(s, selected, self.r + r, actions)
        self.last_node = v
        self.last_action = selected
        v.set_action(selected)
        return v

    def get_last_node_selected(self):
        return self.last_node

    def get_last_action_selected(self):
        return self.last_action

    def update(self, r):
        a = self.get_last_action_selected()
        self.visited += 1
        self.ni[a] += 1
        self.Q[a] += r
        if not self.is_expanded():
            self.expanded = np.all([self.ni[a] > 0 for a in self.actions])

class MCTS:
    def __init__(self, state, ih, horizon=500, limit=5.0, saved=None):
        # print("AQUI NO E")
        self.state = state
        actions = ih.get_actions(state, ih.generate_transition_states(state))
        self.root = Node(state, parent=None, action=None, r=0, actions=actions, ih=ih, nodes={})
        self.ih = ih
        self.horizon = horizon
        self.limit = limit
        self.nodes = {state: self.root}
        self.root.nodes = self.nodes

    def get_best_next_state(self):
        s = self.state
        # print(self, self.root.state, self.root.visited, self.nodes[s] if s in self.nodes else None)
        # print("=======START=============")
        # print(self.state)
        t1 = time.time()
        while True:
            t2 = time.time()
            if t2 - t1 > self.limit: break
            s = self.tree_policy(self.root)
            r = self.default_policy(s)
            self.backpropagate(s.parent, r)
        # print(self.root.get_best_child_duct(0).state)
        # print("========END============")
        return self.root.get_best_child_duct(0).state

    def tree_policy(self, vj):
        count = 1.
        while True:
            vi = vj
            if not vi.is_expanded(): return vi.expand()
            vj = vi.get_best_child_duct(300)
            a = self.ih.get_action(vi.state, vj.state)
            if self.ih.is_terminal(vi.state, vj.state) or count >= self.horizon: break
            count += 1
        return vj

    def default_policy(self, v):
        s_goal = v.state
        r = v.r
        count = 1.
        while True:
            sv = s_goal
            s_goal = self.ih.epsilon_greedy(sv)
            a = self.ih.get_action(sv, s_goal)
            r += (0.9**count)*self.ih.get_reward(sv, a, s_goal) # discounted reward based on amount of actions taken
            if self.ih.is_terminal(sv, s_goal) or count >= self.horizon: break
            count += 1
        return r

    def backpropagate(self, s, r):
        if s is None: return
        s.update(r)
        self.backpropagate(s.parent, r)
