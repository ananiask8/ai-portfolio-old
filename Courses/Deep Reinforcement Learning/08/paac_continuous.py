#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import collections
from torch.distributions import Normal

import continuous_mountain_car_evaluator

class OneHotLike(nn.Module):
    def __init__(self, num_classes):
        super(OneHotLike, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, num_classes=self.num_classes).sum(dim=1).float()

class Network(nn.Module):
    def __init__(self, env, args):
        super(Network, self).__init__()
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]
        self.action_lows, self.action_highs = list(zip(*env.action_ranges))[0]
        self.entropy_regularization = args.entropy_regularization
        # Create `_model`, which: processes `states`. Because `states` are
        # (vectors of) tile indices, you need to convert them to one-hot-like
        # encoding. I.e., for batch example i, state should be a vector of
        # length `weights` with `tiles` ones on indices `states[i,
        # 0..`tiles`-1] and the rest being zeros.
        #
        # The model computes `mus` and `sds`, each of shape [batch_size, action_components].
        # Compute each independently using `states` as input, adding a fully connected
        # layer with args.hidden_layer units and ReLU activation. Then:
        # - For `mus` add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required [-1,1] range, you can apply
        #   `tf.tanh` activation.
        # - For `sds` add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        # The model also computes `values`, starting with `states` and
        # - add a fully connected layer of size args.hidden_layer and ReLU activation
        # - add a fully connected layer with 1 output and no activation
        self._mu_head = nn.Sequential(
            OneHotLike(env.weights),
            nn.Linear(env.weights, args.hidden_layer),
            nn.ReLU(),
            nn.Linear(args.hidden_layer, action_components),
            nn.Hardtanh(min_val=self.action_lows, max_val=self.action_highs)
        )
        self._sd_head = nn.Sequential(
            OneHotLike(env.weights),
            nn.Linear(env.weights, args.hidden_layer),
            nn.ReLU(),
            nn.Linear(args.hidden_layer, action_components),
            nn.Softplus()
        )
        self._value_head = nn.Sequential(
            OneHotLike(env.weights),
            nn.Linear(env.weights, args.hidden_layer),
            nn.ReLU(),
            nn.Linear(args.hidden_layer, 1)
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

    def forward(self, x):
        return self._mu_head(x), self._sd_head(x), self._value_head(x)

    def train(self, states, actions, returns):
        #  Run the model on given states and compute
        # `sds`, `mus` and `values`. Then create `action_distribution` using
        # `Normal` distribution class and computed `mus` and `sds`.
        mus, sds, values = self.forward(states)
        action_dist = Normal(mus, sds)

        # Compute `loss` as a sum of three losses:
        # - negative log probability of the `actions` in the `action_distribution`
        #   (using `log_prob` method). You need to sum the log probabilities
        #   of subactions for a single batch example (using `tf.reduce_sum` with `axis=1`).
        #   Then weight the resulting vector by `(returns - tf.stop_gradient(values))`
        #   and compute its mean.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        # - mean square error of the `returns` and `values`
        loss = -(action_dist.log_prob(actions).sum(dim=1).view(-1, 1) * (returns - values.detach())).mean() \
                -(args.entropy_regularization * action_dist.entropy()).mean() + F.mse_loss(values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def max_actions(self, states):
        mus, _, _ = self.forward(states)
        return mus

    def sample_actions(self, states):
        mus, sds, _ = self.forward(states)
        action_dist = Normal(mus, sds)
        return action_dist.sample().clamp(min=self.action_lows, max=self.action_highs)

    def predict_values(self, states):
        _, _, values = self.forward(states)
        return values

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=128, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")
    parser.add_argument("--workers", default=8, type=int, help="Number of parallel workers.")
    parser.add_argument("--recodex", default=True, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=90, type=int, help="Goal value for task.")
    args = parser.parse_args()

    # Create the environment
    env = continuous_mountain_car_evaluator.environment(tiles=args.tiles)

    # Construct the network
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    network = Network(env, args)
    if args.recodex:
        network.load_state_dict(torch.load("model_paac_cont.py", map_location=device))

    # Initialize parallel workers by env.parallel_init
    batch_states = torch.tensor(env.parallel_init(args.workers), device=device)
    while not args.recodex:
        # Training
        for _ in range(args.evaluate_each):
            # Choose actions using network.predict_actions.
            # using np.random.normal to sample action and np.clip
            # to clip it using action_lows and action_highs,
            batch_actions = network.sample_actions(batch_states).view(-1,1)

            # Perform steps by env.parallel_step
            steps = env.parallel_step(batch_actions.numpy())

            # Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            batch_next_states, batch_rewards, batch_dones, _ = zip(*steps)
            batch_next_states = torch.tensor(batch_next_states, device=device)
            batch_rewards = torch.tensor(batch_rewards, device=device, dtype=torch.float32).view(-1, 1)
            batch_dones = torch.tensor(batch_dones, device=device).view(-1, 1)
            batch_next_states_values = network.predict_values(batch_next_states)
            batch_returns = batch_rewards + args.gamma * batch_next_states_values * ~batch_dones

            # Train network using current states, chosen actions and estimated returns
            network.train(batch_states, batch_actions, batch_returns)
            batch_states = batch_next_states

        torch.save(network.state_dict(), "model_paac_cont.py")
        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action = network.max_actions(torch.tensor([state], device=device)).detach().numpy()[0]
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

            action = network.max_actions(torch.tensor([state], device=device)).detach().numpy()[0]
            state, reward, done, _ = env.step(action)
