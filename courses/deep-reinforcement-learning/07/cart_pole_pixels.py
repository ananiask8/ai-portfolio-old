#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import collections
from torch.distributions import Categorical

import cart_pole_pixels_evaluator

class Network(nn.Module):
    def __init__(self, env, args):
        super(Network, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 4, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(80, kernel_size=5, stride=2), kernel_size=2, stride=2), kernel_size=5, stride=2), kernel_size=2, stride=2), kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(80, kernel_size=5, stride=2), kernel_size=2, stride=2), kernel_size=5, stride=2), kernel_size=2, stride=2), kernel_size=3, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(convw * convh * 1, 5000),
            nn.Dropout(p=0.6)
        )
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
            value_losses.append(F.smooth_l1_loss(v, G))
            policy_losses.append(-(delta * log_probs).sum())
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc(out))
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
    parser.add_argument("--batch_size", default=64, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=300, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--recodex", default=False, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=450, type=int, help="Goal value for task.")
    parser.add_argument("--period", default=100, type=int, help="Period to evaluate goal value for task.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    network = Network(env, args).to(device)
    if args.recodex:
        network.load_state_dict(torch.load("model_reinforce_pixels.py", map_location=device))

    # Training
    SavedAction = collections.namedtuple('SavedAction', ['log_prob', 'action'])
    evaluating = args.recodex
    episode_returns, episodes_count = [], 0
    eps_zero = np.finfo(np.float32).eps.item()
    while not evaluating:
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            S, A, R = [], [], [None]
            state, done, episode_return = env.reset().transpose((2, 0, 1)).astype(np.float32), False, 0
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()
                action, log_prob = network.sample_action(torch.tensor([state], device=device))
                next_state, reward, done, _ = env.step(action.item())
                next_state = next_state.transpose((2, 0, 1)).astype(np.float32)
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
        torch.save(network.state_dict(), "model_reinforce_pixels.py")

    # Final evaluation
    while True:
        state, done = env.reset(True).transpose((2, 0, 1)).astype(np.float32), False
        while not done:
            # Compute action `probabilities` using `network.predict` and current `state`
            policy_probs, _ = network(torch.tensor([state], device=device))

            # Choose greedy action this time
            action = torch.argmax(policy_probs)
            state, reward, done, _ = env.step(action.item())
            state = state.transpose((2, 0, 1)).astype(np.float32)
