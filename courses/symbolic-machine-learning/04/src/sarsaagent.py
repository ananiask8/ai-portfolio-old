import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from carddeck import *
import numpy as np
from random import random, randint

class SarsaAgent:
    '''
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate).
    '''
    HAND = {
        "HAS_11": 2,
        "HAND_VALUE": 32
    }
    HISTORY = {
        "HALVES": {"|C|": 42, "MIN": -14, "MAX": 22, "STEP": 0.5, "VALUES": [0, 0, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1, -1]},
        "HiLo": {"|C|": 16, "MIN": -14, "MAX": 14, "STEP": 1, "VALUES": [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1]} 
    }
    ACTIONS = [0, 1]

    def __init__(self, env, number_of_epochs):
        self.env = env
        self.number_of_epochs = number_of_epochs
        # self.Ne, self.R_opt = 5, 1
        self.Ne, self.R_opt = 100, 1
        self.c, self.discount_factor = 0.8, 0.8
        self.config = {"HISTORY": None, "HAS_11": True}

    def train(self):
        Q, Ns = self.init_state_space()
        Q_16_11_hit = [0] * self.number_of_epochs
        Q_16_11_stay = [0] * self.number_of_epochs
        for i in range(self.number_of_epochs):
            observation = self.env.reset()
            terminal = False
            reward = 0
            new_feature = self.get_features(observation)
            while not terminal:
                old_feature = new_feature
                old_action, old_q = self.select_action(old_feature, Q, Ns)
                Ns[old_feature + old_action] += 1
                observation, reward, terminal, _ = self.env.step(old_action[0])
                new_feature = self.get_features(observation)
                new_action, new_q = self.select_action(new_feature, Q, Ns)
                Q[old_feature + old_action] = Q[old_feature + old_action] + self.alpha(Ns[old_feature + old_action])*(reward + self.discount_factor*new_q - Q[old_feature + old_action])
            Q_16_11_stay[i] = Q[(True, 11+5, True, 11, 0)]
            Q_16_11_hit[i] = Q[(True, 11+5, True, 11, 1)]
        self.plot_series(Q_16_11_hit, "Q_16_11_hit.png")
        self.plot_series(Q_16_11_stay, "Q_16_11_stay.png")
            #self.env.render()
        # self.print_Q(Q)
        # print((False, 9+10+2, False, 4, 1), Q[(False, 10+9+2, False, 4, 1)])
        # print((True, 11+5, True, 11, 0), Q[(True, 11+5, True, 11, 0)])
        # print((True, 11+5, True, 11, 1), Q[(True, 11+5, True, 11, 1)])

    def init_state_space(self):
        Q, Ns = {}, {}
        h = self.config["HISTORY"]
        if self.config["HISTORY"] is not None:
            for c in np.arange(self.HISTORY[h]["MIN"], self.HISTORY[h]["MAX"], self.HISTORY[h]["STEP"]):
                for v in range(self.HAND["HAND_VALUE"]):
                    for w in range(self.HAND["HAND_VALUE"]):
                        for a in self.ACTIONS:
                            if self.config["HAS_11"]:
                                for v_has_11 in range(2):
                                    for w_has_11 in range(2):
                                        Q[(c, v_has_11, v, w_has_11, w, a)] = 0
                                        Ns[(c, v_has_11, v, w_has_11, w, a)] = 0
                            else:
                                Q[(c, v, w, a)] = 0
                                Ns[(c, v, w, a)] = 0
        else:
            for v in range(self.HAND["HAND_VALUE"]):
                    for w in range(self.HAND["HAND_VALUE"]):
                        for a in self.ACTIONS:
                            if self.config["HAS_11"]:
                                for v_has_11 in range(2):
                                    for w_has_11 in range(2):
                                        Q[(v_has_11, v, w_has_11, w, a)] = 0
                                        Ns[(v_has_11, v, w_has_11, w, a)] = 0
                            else:
                                Q[(v, w, a)] = 0
                                Ns[(v, w, a)] = 0
        return Q, Ns

    def get_features(self, observation):
        feature = ()
        if self.config["HISTORY"] is not None: feature += (self.get_history(observation),)
        if self.config["HAS_11"]: feature += (self.has11(observation.player_hand),)
        feature += (observation.player_hand.value(),)
        if self.config["HAS_11"]: feature += (self.has11(observation.dealer_hand),)
        feature += (observation.dealer_hand.value(),)
        return feature

    def get_history(self, observation):
        values = self.HISTORY[self.config["HISTORY"]]["VALUES"]
        return sum([values[card.value()] for card in observation.player_hand.cards]) + sum([values[card.value()] for card in observation.dealer_hand.cards])

    def has11(self, hand):
        return 1 if sum([card.value() for card in hand.cards]) < hand.value() else 0

    def print_Q(self, Q):
        for key, val in Q.items():
            print(str((key, val)))

    def select_action(self, feature, Q, Ns):
        sample = random()
        visits = sum([Ns[feature + (a_p,)] for a_p in self.ACTIONS])
        eps = 1/visits if visits > 0 else 1
        if sample > eps: return self.argmax_action(feature, Q, Ns)
        else:
            a = self.ACTIONS[randint(0, len(self.ACTIONS) - 1)]
            return (a,), Q[feature + (a,)]

    def argmax_action(self, feature, Q, Ns):
        a, max_a = 0, -float("inf")
        for a_p in self.ACTIONS:
            u = self.exploration_function(Q[feature + (a_p,)], Ns[feature + (a_p,)])
            if u > max_a: max_a, a = u, a_p
        return (a,), max_a
 
    def alpha(self, count):
        return self.c/(self.c - 1 + count)
 
    def exploration_function(self, u, n):
        if n < self.Ne: return self.R_opt
        return u

    def plot_series(self, arr, fileName = None):
        plt.clf()
        plt.plot(arr)
        if fileName is not None:
            plt.savefig(fileName)

