#!/usr/bin/env python3
import argparse
import sys

import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}

class BaseAgent():
    def __init__(self, args):
        self._generator = np.random.RandomState(args.seed)
        self._bandits = args.bandits
        self._epsilon = args.epsilon
        self._alpha = args.alpha
        self._c = args.c
        self._Q = np.array(list(map(float, [args.initial]*self._bandits)))
        self._N = np.array([0]*self._bandits)
        self._t = 0
        self._H = np.array(list(map(float, [args.initial]*self._bandits)))

    def select_action(self):
        pass

    def update(self):
        pass

class EpsGreedyAgent(BaseAgent):
    def select_action(self):
        x = self._generator.uniform(0, 1)
        return self._generator.randint(0, self._bandits) if x < self._epsilon else np.argmax(self._Q)

    def update(self, action, reward):
        self._t += 1
        self._N[action] += 1
        lr = self._alpha if self._alpha > 0 else 1./self._N[action]
        self._Q[action] += lr*(reward - self._Q[action]) 

class UcbAgent(BaseAgent):
    def select_action(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            exploration_factor = self._c*np.sqrt(np.log(self._t) / self._N)
        ucb = self._Q + exploration_factor
        return np.argmax(ucb)

    def update(self, action, reward):
        self._t += 1
        self._N[action] += 1
        lr = self._alpha if self._alpha > 0 else 1./self._N[action]
        self._Q[action] += lr*(reward - self._Q[action]) 

class GradientAgent(BaseAgent):
    def select_action(self):
        self._pi_dist = np.exp(self._H)/np.sum(np.exp(self._H))
        return self._generator.choice(self._bandits, 1, p=self._pi_dist)[0]

    def update(self, action, reward):
        self._t += 1
        self._N[action] += 1
        self._Q[action] += (reward - self._Q[action]) / self._N[action]
        self._H += self._alpha*(reward - self._Q)*((action == self._bandits) - self._pi_dist)

def factory(mode):
    return {
        "greedy": EpsGreedyAgent,
        "ucb": UcbAgent,
        "gradient": GradientAgent
    }[mode]


parser = argparse.ArgumentParser()
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
parser.add_argument("--c", default=1, type=float, help="Confidence level in ucb (if applicable).")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial value function levels (if applicable).")

def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)

    all_avg_rewards = []
    for episode in range(args.episodes):
        env.reset()
        agent = factory(args.mode)(args)
        rewards = []
        done = False
        while not done:
            # Action selection according to mode
            action = agent.select_action()
            _, reward, done, _ = env.step(action)
            # Update parameters
            agent.update(action, reward)
            rewards.append(reward)
        all_avg_rewards.append(np.mean(np.array(rewards)))
    # For every episode, compute its average reward (a single number),
    # obtaining `args.episodes` values. Then return the final score as
    # mean and standard deviation of these `args.episodes` values.
    r = np.array(all_avg_rewards)
    return np.mean(r), np.std(r)

if __name__ == "__main__":
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print("{:.2f} {:.2f}".format(mean, std))
