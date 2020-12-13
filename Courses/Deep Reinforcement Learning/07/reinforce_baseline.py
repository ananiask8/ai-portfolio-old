#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import collections
from torch.distributions import Categorical

import cart_pole_evaluator

class Network(nn.Module):
    def __init__(self, env, args):
        super(Network, self).__init__()
        self.fc = nn.Linear(4, 5000)
        self.policy_head = nn.Linear(5000, env.actions)
        self.value_head = nn.Linear(5000, 1)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

    def train(self, batch_states, batch_actions, batch_returns):
        policy_losses, value_losses = [], []
        for i in range(len(batch_states)):
            S, A, G = batch_states[i], batch_actions[i], batch_returns[i][:-1]
            log_probs, actions = map(torch.stack, zip(*A))
            _, v = self.forward(S)
            G = G.view(-1, 1)
            delta = (G - v).detach()
            value_losses.append(-(delta * F.smooth_l1_loss(v, G)).sum())
            # value_losses.append(-(delta * v).sum())
            policy_losses.append(-(delta * log_probs).sum())
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        out = F.relu(self.fc(x))
        policy_probs = F.softmax(self.policy_head(out), dim=-1)
        v = self.value_head(out)
        return policy_probs, v

    def sample_action(self, x):
        policy_probs, _ = self.forward(x)
        policy_dist = Categorical(policy_probs)
        action = policy_dist.sample()
        return action, policy_dist.log_prob(action)


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=300, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--recodex", default=True, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=490, type=int, help="Goal value for task.")
    parser.add_argument("--period", default=100, type=int, help="Period to evaluate goal value for task.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    network = Network(env, args).to(device)
    if args.recodex:
        network.load_state_dict(torch.load("model_reinforce_baseline.py", map_location=device))

    # Training
    SavedAction = collections.namedtuple('SavedAction', ['log_prob', 'action'])
    evaluating = args.recodex
    episode_returns, episodes_count = [], 0
    eps_zero = np.finfo(np.float32).eps.item()
    while not evaluating and not episodes_count > args.episodes:
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            S, A, R = [], [], [None]
            state, done, episode_return = env.reset().astype(np.float32), False, 0
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()
                action, log_prob = network.sample_action(torch.tensor([state], device=device))
                next_state, reward, done, _ = env.step(action.item())
                next_state = next_state.astype(np.float32)
                S += [state]
                A += [SavedAction(log_prob, action)]
                R += [reward]
                state = next_state
                episode_return += reward

            # Compute returns by summing rewards (with discounting)
            T = len(R) - 1
            G = [0]*len(R)
            for i in reversed(range(T)):
                G[i] = R[i+1] + args.gamma*G[i+1]
            S = torch.tensor(S, device=device)
            G = torch.tensor(G, device=device)

            # Decide if we want to start evaluating
            episodes_count += 1
            print("Episode #{}: {}".format(episodes_count, episode_return))
            episode_returns = episode_returns[-(args.period - 1):] + [episode_return]
            evaluating |= np.mean(episode_returns) > args.goal

            # Add states, actions and returns to the training batch
            batch_states += [S]
            batch_actions += [A]
            batch_returns += [G]

        # Train using the generated batch
        network.train(batch_states, batch_actions, batch_returns)
        torch.save(network.state_dict(), "model_reinforce_baseline.py")

    # Final evaluation
    while True:
        state, done = env.reset(True).astype(np.float32), False
        while not done:
            # Compute action `probabilities` using `network.predict` and current `state`
            policy_probs, _ = network(torch.tensor([state], device=device))

            # Choose greedy action this time
            action = torch.argmax(policy_probs)
            state, reward, done, _ = env.step(action.item())
            state = state.astype(np.float32)
