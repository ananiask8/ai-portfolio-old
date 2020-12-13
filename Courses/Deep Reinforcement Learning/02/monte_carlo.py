#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

def eps_soft_sample(nA, a, eps):
    eps = 0 if eps is None else eps
    dist = [eps / nA] * nA
    dist[a] += 1 - eps
    return np.random.choice(nA, 1, p=dist)[0]

def argmax(f):
    return max(range(len(f)), key=lambda x: f[x])

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.15, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # TODO: Implement Monte-Carlo RL algorithm.
    q_function = [[0. for i in range(env.actions)] for j in range(env.states)]

    # The overall structure of the code follows.
    #
    # First visit
    # policy = [0] * env.states
    # returns = [[[] for i in range(env.actions)] for j in range(env.states)]
    # for i in range(args.episodes):
    #     # Perform a training episode
    #     episode = []
    #     T = 0
    #     state, done = env.reset(), False
    #     while not done:
    #         # if args.render_each and env.episode and env.episode % args.render_each == 0:
    #         #     env.render()
    #         action = eps_soft_sample(env.actions, policy[state], args.epsilon)
    #         next_state, reward, done, _ = env.step(action)
    #         episode += map(int, [state, action, reward])
    #         state = next_state
    #         T += 1
    #
    #     G = 0
    #     for t in reversed(range(T)):
    #         step =  { "s_t": episode[3 * t], "a_t": episode[3 * t + 1], "r_t+1": episode[3 * t + 2] }
    #         prev_steps = episode[:(3 * t)]
    #
    #         G = args.gamma * G + step["r_t+1"]
    #         if "{},{}".format(step["s_t"], step["a_t"]) in ",".join(map(str,prev_steps)): 
    #             returns[step["s_t"]][step["a_t"]].append(G)
    #             q_function[step["s_t"]][step["a_t"]] = np.mean(returns[step["s_t"]][step["a_t"]])
    #             policy[step["s_t"]] = argmax(q_function[step["s_t"]])

    # Every visit
    C = [[0 for i in range(env.actions)] for j in range(env.states)]
    for i in range(args.episodes):
        # Perform a training episode
        episode = []
        T = 0
        state, done = env.reset(), False
        eps_i = (args.epsilon_final - args.epsilon) / args.episodes * i + args.epsilon
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = eps_soft_sample(env.actions, argmax(q_function[state]), eps_i)
            next_state, reward, done, _ = env.step(action)
            episode += [state, action, reward]
            state = next_state
            T += 1

        G = 0
        for t in reversed(range(T)):
            step =  { "s_t": episode[3 * t], "a_t": episode[3 * t + 1], "r_t+1": episode[3 * t + 2] }
            G = args.gamma * G + step["r_t+1"]
            C[step["s_t"]][step["a_t"]] += 1
            q_function[step["s_t"]][step["a_t"]] += (G - q_function[step["s_t"]][step["a_t"]]) / C[step["s_t"]][step["a_t"]]

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = eps_soft_sample(env.actions, argmax(q_function[state]), 0)
            state, reward, done, _ = env.step(action)
    env.close()
