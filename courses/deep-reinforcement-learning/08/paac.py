#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import collections
from torch.distributions import Categorical

import gym_evaluator

class PolicyNetwork(nn.Module):
    def __init__(self, env, args):
        super(PolicyNetwork, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(4, args.hidden_layer),
            nn.ReLU(),
            nn.Linear(args.hidden_layer, env.actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

    def train(self, batch_values, batch_log_probs, batch_returns):
        delta = (batch_returns - batch_values).detach()
        loss = -(delta * batch_log_probs).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        return self.nn(x)

    def sample_action(self, x):
        policy_probs = self.forward(x)
        policy_dist = Categorical(policy_probs)
        action = policy_dist.sample()
        return action, policy_dist.log_prob(action)

class ValueNetwork(nn.Module):
    def __init__(self, env, args):
        super(ValueNetwork, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(4, args.hidden_layer),
            nn.ReLU(),
            nn.Linear(args.hidden_layer, 1)
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

    def train(self, batch_states, batch_returns):
        batch_values = self.forward(batch_states)
        loss = F.mse_loss(batch_values, batch_returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        return self.nn(x)

class Network:
    def __init__(self, env, args, device):
        # Similarly to reinforce, define two models:
        # - _policy, which predicts distribution over the actions
        # - _value, which predicts the value function
        # Use independent networks for both of them, each with
        # `args.hidden_layer` neurons in one hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        self._policy = PolicyNetwork(env, args).to(device)
        self._value = ValueNetwork(env, args).to(device)
        if args.recodex:
            self._policy.load_state_dict(torch.load("model_paac_policy.py", map_location=device))
            self._value.load_state_dict(torch.load("model_paac_value.py", map_location=device))

    def train(self, batch_states, batch_actions, batch_log_probs, batch_returns):
        # Train the policy network using policy gradient theorem
        # and the value network using MSE.
        self._policy.train(self._value(batch_states).detach(), batch_log_probs, batch_returns)
        self._value.train(batch_states, batch_returns)

    def predict_actions(self, states):
        return self._policy(states)

    def sample_actions(self, states):
        return self._policy.sample_action(states)

    def predict_values(self, states):
        return self._value(states)

    def save(self):
        torch.save(self._policy.state_dict(), "model_paac_policy.py")
        torch.save(self._value.state_dict(), "model_paac_value.py")

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=128, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--workers", default=8, type=int, help="Number of parallel workers.")
    parser.add_argument("--recodex", default=True, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=450, type=int, help="Goal value for task.")
    args = parser.parse_args()

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    network = Network(env, args, device)

    # Initialize parallel workers by env.parallel_init
    batch_states = torch.tensor(env.parallel_init(args.workers), device=device, dtype=torch.float32)
    while not args.recodex:
        # Training
        for _ in range(args.evaluate_each):
            # Choose actions using network.predict_actions
            batch_actions, batch_log_probs = network.sample_actions(batch_states)

            # Perform steps by env.parallel_step
            steps = env.parallel_step(batch_actions.numpy())

            # Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            batch_next_states, batch_rewards, batch_dones, _ = zip(*steps)
            batch_next_states, batch_rewards = map(lambda p: torch.tensor(p, device=device, dtype=torch.float32), [batch_next_states, batch_rewards])
            batch_dones = torch.tensor(batch_dones, device=device)
            batch_next_states_values = network.predict_values(batch_next_states)

            batch_returns = batch_rewards.view(-1, 1) + args.gamma * batch_next_states_values * ~batch_dones.view(-1, 1)

            # Train network using current states, chosen actions and estimated returns
            network.train(batch_states, batch_actions.view(-1,1), batch_log_probs, batch_returns)
            batch_states = batch_next_states
        network.save()

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict_actions(torch.tensor([state], device=device, dtype=torch.float32))
                action = torch.argmax(probabilities).item()
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))
        if np.mean(returns) >= args.goal:
            break

    # On the end perform final evaluations with `env.reset(True)`
    while True:
        state, done = env.reset(True), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            probabilities = network.predict_actions(torch.tensor([state], device=device, dtype=torch.float32))
            action = torch.argmax(probabilities).item()
            state, reward, done, _ = env.step(action)
