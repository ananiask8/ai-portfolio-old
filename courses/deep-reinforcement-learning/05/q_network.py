#!/usr/bin/env python3
import collections
import itertools

import numpy as np
import torch
import torch.nn as nn

import cart_pole_evaluator

class Network(nn.Module):
    def __init__(self, env, args):
        # Define a training method. Generally you have two possibilities
        # - pass new q_values of all actions for a given state; all but one are the same as before
        # - pass only one new q_value for a given state, including the index of the action to which
        #   the new q_value belongs
        super(Network, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(env.state_shape[0], args.hidden_layer_size), nn.LeakyReLU(negative_slope=0.01))])
        self.hidden_layers.extend([nn.Sequential(nn.Linear(args.hidden_layer_size, args.hidden_layer_size), nn.LeakyReLU(negative_slope=0.01)) for i in range(args.hidden_layers - 1)])
        self.hidden_layers.extend([nn.Sequential(nn.Linear(args.hidden_layer_size, env.actions))])
        [layer.apply(self._init_weights) for layer in self.hidden_layers]
        self.optimizer = torch.optim.RMSprop(params=self.parameters(), lr=args.learning_rate)
        self.huber = nn.SmoothL1Loss()
        self.gamma = args.gamma

    def forward(self, x):
        out = x
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
        return out

    def replay(self, states, actions, rewards, dones, next_states, target):
        target.eval()
        q_next = target.forward(next_states)
        argmax_actions = torch.argmax(q_next, dim=1)
        y = rewards.view(-1, 1) + self.gamma * q_next.gather(1, argmax_actions.view(-1, 1)) * ~dones.view(-1, 1)
        y = y.detach()
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            q_state = self.forward(states)
            # print(q_state)
            loss = self.huber(q_state.gather(1, actions.view(-1, 1)), y)
            # print(loss)
            loss.backward()
            self.optimizer.step()

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, a=0.01)
            if m.bias is not None:
                m.bias.data.zero_()

def eps_soft_sample(nA, a, eps):
    eps = 0 if eps is None else eps
    dist = [eps / nA] * nA
    dist[a] += 1 - eps
    return np.random.choice(nA, 1, p=dist)[0]

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--recodex", default=True, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=400, type=int, help="Goal value for task.")
    parser.add_argument("--period", default=100, type=int, help="Period to evaluate goal value for task.")
    parser.add_argument("--update_freq", default=10000, type=int, help="Frequency of updates to target network.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    # np.random.seed(42)
    # torch.random.manual_seed(42)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args).double()
    if args.recodex:
        network.load_state_dict(torch.load("model_q_network.py"))
    target = Network(env, args).double()
    target.load_state_dict(network.state_dict())

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    evaluating = args.recodex
    epsilon = args.epsilon
    episode_returns, count = [], 0
    while not evaluating:
        # Perform episode
        state, done, episode_return = env.reset(), False, 0
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # Compute action using epsilon-greedy policy. You can compute
            # the q_values of a given state using
            with torch.no_grad():
                q_state = network(torch.tensor(state))
            action = eps_soft_sample(env.actions, torch.argmax(q_state).item(), epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            count += 1

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))
            # replay_buffer = replay_buffer[-1000:] if no maxlen set

            # If the replay_buffer is large enough, preform a training batch
            # of `args.batch_size` uniformly randomly chosen transitions.
            idx = np.random.randint(low=0, high=len(replay_buffer), size=args.batch_size)
            t = [replay_buffer[i] for i in idx]
            states, actions, rewards, dones, next_states = map(torch.tensor, zip(*t))
            network.replay(states, actions, rewards, dones, next_states, target)
            state = next_state
            if count % args.update_freq == 0:
                target.load_state_dict(network.state_dict())
                # input("TYPE SOMETHING")

        # Decide if we want to start evaluating
        episode_returns = episode_returns[-(args.period - 1):] + [episode_return]
        evaluating |= np.mean(episode_returns) > args.goal
        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        else:
            torch.save(network.state_dict(), "model_q_network.py")

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                # env.render()
            action = torch.argmax(network.forward(torch.tensor(state))).item()
            state, reward, done, _ = env.step(action)
