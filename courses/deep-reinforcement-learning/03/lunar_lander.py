#!/usr/bin/env python3
import numpy as np
import multiprocessing

import lunar_lander_evaluator

def eps_soft_sample(nA, a, eps):
    eps = 0 if eps is None else eps
    dist = [eps / nA] * nA
    dist[a] += 1 - eps
    return np.random.choice(nA, 1, p=dist)[0]

def eps_soft_dist(nA, a, eps):
    eps = 0 if eps is None else eps
    dist = [eps / nA] * nA
    dist[a] += 1 - eps
    return dist

def argmax(f):
    return max(range(len(f)), key=lambda x: f[x])

# def learn_from_expert(env, args):
#     # Every-visit Monte Carlo
#     Q = [[0. for i in range(env.actions)] for j in range(env.states)]
#     C = [[0 for i in range(env.actions)] for j in range(env.states)]
#     for i in range(args.expert_episodes):
#         print(i)
#         state, trajectory = env.expert_trajectory()
#         trajectory = np.concatenate(([state], np.array(trajectory).flatten()[:-1])).reshape(-1,3)
#         G = 0
#         for state, action, reward in trajectory:
#             state, action = int(state), int(action)
#             G = args.gamma * G + reward
#             C[state][action] += 1
#             Q[state][action] += (G - Q[state][action]) / C[state][action]
#     return Q

# def learn_from_expert(env, args):
#     # n-step Double Q-learning
#     Q1 = np.array([[0. for i in range(env.actions)] for j in range(env.states)])
#     Q2 = Q1.copy()
#     Q = (Q1, Q2)
#     C = [[0 for i in range(env.actions)] for j in range(env.states)]
#     for i in range(args.expert_episodes):
#         # print(i)
#         eps_i = (args.epsilon_final - args.epsilon) / args.expert_episodes * i + args.epsilon
#         alpha_i = (args.alpha_final - args.alpha) / args.expert_episodes * i + args.alpha

#         # Perform a training episode
#         state, trajectory = env.expert_trajectory()
#         trajectory = np.concatenate(([state], np.array(trajectory).flatten()[:-1])).reshape(-1,3)
#         S, A, R = zip(*trajectory)
#         S, A, R = list(map(int, S)), list(map(int, A)), [None] + list(R)
#         tau, t, T = 0, 0, len(S)
#         while not tau == T - 1:
#             Q = (Q1, Q2) if np.random.binomial(1, 0.5) == 0 else (Q2, Q1)
#             tau = t - args.n_steps + 1
#             if tau >= 0:
#                 min_i, max_i = tau + 1, min(tau + args.n_steps, T)
#                 G = np.array([args.gamma ** (i - tau - 1) * R[i] for i in range(min_i, max_i)]).sum()
#                 if tau + args.n_steps < T:
#                     G += (args.gamma ** args.n_steps) * Q[1][ S[tau + args.n_steps] ][ np.argmax(Q[0][S[tau + args.n_steps]]) ]
#                 Q[0][S[tau], A[tau]] += alpha_i * (G - Q[0][S[tau], A[tau]])
#             t += 1
#     Q = (Q[0] + Q[1]).tolist()
#     return Q

def learn_from_expert(env, args):
    # n-step Tree Backup
    Q = np.array([[0. for i in range(env.actions)] for j in range(env.states)])
    for i in range(args.expert_episodes):
        print(i)
        eps_i = 0
        alpha_i = (args.alpha_final - args.alpha) / args.expert_episodes * i + args.alpha

        # Perform a training episode
        state, trajectory = env.expert_trajectory()
        trajectory = np.concatenate(([state], np.array(trajectory).flatten()[:-1])).reshape(-1,3)
        S, A, R = zip(*trajectory)
        S, A, R = list(map(int, S)), list(map(int, A)), [None] + list(R)
        tau, t, T = 0, 0, len(S)
        while not tau == T - 1:
            tau = t - args.n_steps + 1
            if tau >= 0:
                if t + 1 >= T:
                    G = R[T]
                else:
                    dist_s = eps_soft_dist(env.actions, np.argmax(Q[S[t + 1]]), eps_i)
                    G = R[t + 1] + args.gamma * np.sum(np.array(dist_s) * Q[S[t + 1]])
                for k in reversed(range(tau + 1, min(t, T - 1) + 1)):
                    dist_s = eps_soft_dist(env.actions, np.argmax(Q[S[k]]), eps_i)
                    dist_not_Ak = dist_s[:A[k]] + dist_s[(A[k] + 1):]
                    Q_not_Ak = np.concatenate((Q[S[k]][:A[k]], Q[S[k]][(A[k] + 1):]))
                    G = R[k] + args.gamma * np.sum(dist_not_Ak * Q_not_Ak) + args.gamma * dist_s[A[k]] * G
                Q[S[tau], A[tau]] += alpha_i * (G - Q[S[tau], A[tau]])
            t += 1
    return Q.tolist()

# def train(env, args):
#     # n-step Double Q-learning
#     Q1 = np.array(learn_from_expert(env, args))
#     Q2 = Q1.copy()
#     for i in range(args.episodes):
#         # eps_i = (args.epsilon_final - args.epsilon) / args.episodes * i + args.epsilon
#         eps_i = args.epsilon_final
#         # alpha_i = (args.alpha_final - args.alpha) / args.episodes * i + args.alpha
#         alpha_i = args.alpha_final

#         # Perform a training episode
#         state, done = env.reset(), False
#         S, A, R = [state], [eps_soft_sample(env.actions, np.argmax(Q1[state] + Q2[state]), eps_i)], [None]
#         tau, t, T = 0, 0, float("inf")
#         while not tau == T - 1:
#             Q = (Q1, Q2) if np.random.binomial(1, 0.5) == 0 else (Q2, Q1)
#             if args.render_each and env.episode and env.episode % args.render_each == 0:
#                 env.render()
#             if t < T:
#                 next_state, reward, done, _ = env.step(A[t])
#                 S += [next_state]
#                 R += [reward]
#                 if done:
#                     T = t + 1
#                 else:
#                     A += [eps_soft_sample(env.actions, np.argmax(Q[0][state] + Q[1][state]), eps_i)]
#             tau = t - args.n_steps + 1
#             if tau >= 0:
#                 min_i, max_i = tau + 1, min(tau + args.n_steps, T)
#                 G = np.array([args.gamma ** (i - tau - 1) * R[i] for i in range(min_i, max_i)]).sum()
#                 if tau + args.n_steps < T:
#                     G += (args.gamma ** args.n_steps) * Q[1][ S[tau + args.n_steps] ][ np.argmax(Q[0][S[tau + args.n_steps]]) ]
#                 Q[0][S[tau], A[tau]] += alpha_i * (G - Q[0][S[tau], A[tau]])
#             t += 1
#             state = next_state
#     Q = (Q[0] + Q[1]).tolist()
#     np.savetxt("lunar_lander_policy_double.py", Q)
#     return Q

def train(env, args):
    # n-step Tree Backup
    # Q = np.loadtxt("lunar_lander_policy.py")
    Q = np.array(learn_from_expert(env, args))
    for i in range(args.episodes):
        # eps_i = (args.epsilon_final - args.epsilon) / args.episodes * i + args.epsilon
        eps_i = args.epsilon_final
        # alpha_i = (args.alpha_final - args.alpha) / args.episodes * i + args.alpha
        alpha_i = args.alpha_final

        # Perform a training episode
        state, done = env.reset(), False
        S, A, R = [state], [eps_soft_sample(env.actions, np.argmax(Q[state]), eps_i)], [None]
        tau, t, T = 0, 0, float("inf")
        while not tau == T - 1:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            if t < T:
                next_state, reward, done, _ = env.step(A[t])
                S += [next_state]
                R += [reward]
                if done:
                    T = t + 1
                else:
                    A += [eps_soft_sample(env.actions, np.argmax(Q[state]), eps_i)]
            tau = t - args.n_steps + 1
            if tau >= 0:
                if t + 1 >= T:
                    G = R[T]
                else:
                    dist_s = eps_soft_dist(env.actions, np.argmax(Q[S[t + 1]]), eps_i)
                    G = R[t + 1] + args.gamma * np.sum(np.array(dist_s) * Q[S[t + 1]])
                for k in reversed(range(tau + 1, min(t, T - 1) + 1)):
                    dist_s = eps_soft_dist(env.actions, np.argmax(Q[S[k]]), eps_i)
                    dist_not_Ak = dist_s[:A[k]] + dist_s[(A[k] + 1):]
                    Q_not_Ak = np.concatenate((Q[S[k]][:A[k]], Q[S[k]][(A[k] + 1):]))
                    G = R[k] + args.gamma * np.sum(dist_not_Ak * Q_not_Ak) + args.gamma * dist_s[A[k]] * G
                Q[S[tau], A[tau]] += alpha_i * (G - Q[S[tau], A[tau]])
            t += 1
            state = next_state
    np.savetxt("lunar_lander_policy.py", Q)
    return Q

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(40)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_episodes", default=60000, type=int, help="Expert training episodes.")
    parser.add_argument("--episodes", default=60000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--n_steps", default=16, type=int, help="Perform n steps in n-step methods.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.05, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--recodex", default=True, type=float, help="Running in ReCodEx?")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # Implement a suitable RL algorithm.
    Q = np.loadtxt("lunar_lander_policy.py") if args.recodex else train(env, args)

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = argmax(Q[state])
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            state, reward, done, _ = env.step(action)

    print("50/50")