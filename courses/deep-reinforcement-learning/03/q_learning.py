#!/usr/bin/env python3
import numpy as np
import time

import mountain_car_evaluator

def eps_soft_sample(nA, a, eps):
    eps = 0 if eps is None else eps
    dist = [eps / nA] * nA
    dist[a] += 1 - eps
    return np.random.choice(nA, 1, p=dist)[0]

def argmax(f):
    return max(range(len(f)), key=lambda x: f[x])

def train(env, args):
    q_function = [[0. for i in range(env.actions)] for j in range(env.states)]
    for i in range(args.episodes):
        eps_i = (args.epsilon_final - args.epsilon) / args.episodes * i + args.epsilon
        alpha_i = (args.alpha_final - args.alpha) / args.episodes * i + args.alpha
        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = eps_soft_sample(env.actions, argmax(q_function[state]), eps_i)
            next_state, reward, done, _ = env.step(action)
            q_function[state][action] += alpha_i * (reward + args.gamma * max(q_function[next_state]) - q_function[state][action])
            state = next_state
    np.savetxt("policy_qlearning.py", q_function)
    return q_function

if __name__ == "__main__":
    # Fix random seed
    # np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.02, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.5, type=float, help="Discounting factor.")
    parser.add_argument("--recodex", default=True, type=float, help="Running in ReCodEx?")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    # Implement Q-learning RL algorithm.
    q_function = np.loadtxt("policy_qlearning.py") if args.recodex else train(env, args)
    
    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = argmax(q_function[state])
            state, reward, done, _ = env.step(action)
