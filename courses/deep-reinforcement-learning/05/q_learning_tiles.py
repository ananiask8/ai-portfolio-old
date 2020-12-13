#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator

def eps_soft_sample(nA, a, eps):
    eps = 0 if eps is None else eps
    dist = [eps / nA] * nA
    dist[a] += 1 - eps
    return np.random.choice(nA, 1, p=dist)[0]

def argmax(f):
    return max(range(len(f)), key=lambda x: f[x])

def train(env, args):
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    evaluating = False
    episode_returns = []
    while not evaluating:
        # Perform a training episode
        state, done, episode_return = env.reset(evaluating), False, 0
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            q_s = sum(W[state])
            action = eps_soft_sample(env.actions, argmax(q_s), epsilon)
            next_state, reward, done, _ = env.step(action)
            q_next = sum(W[next_state])
            W[state, action] += alpha * (reward + args.gamma * max(q_next) - q_s[action])
            # print(alpha * (reward + args.gamma * max(q_next) - q_s[action]))
            # print(alpha, reward, args.gamma, max(q_next), q_s[action])
            # print(W[state, action])
            state = next_state
            episode_return += reward
            if done:
                break

        # Decide if we want to start evaluating
        episode_returns = episode_returns[-(args.period - 1):] + [episode_return]
        evaluating |= np.mean(episode_returns) > args.goal
        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

    np.savetxt("policy_q_learning_tiles.py", W)
    return W

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.02, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    parser.add_argument("--recodex", default=True, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=-100, type=int, help="Goal value for task.")
    parser.add_argument("--period", default=100, type=int, help="Period to evaluate goal value for task.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.loadtxt("policy_q_learning_tiles.py") if args.recodex else train(env, args)

    # Perform the final evaluation episodes
    while True:
        state, done = env.reset(True), False
        while not done:
            # Choose action as a greedy action
            action = argmax(sum(W[state]))
            state, reward, done, _ = env.step(action)
