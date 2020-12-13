#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import collections
from torch.distributions import Normal

import gym_evaluator

class SigmoidRescaled(nn.Module):
    def __init__(self, ranges):
        super(SigmoidRescaled, self).__init__()
        self.min, self.max = ranges
        self.components = len(self.min)

    def forward(self, x):
        return torch.sigmoid(x)*(self.max - self.min) + self.min

class PolicyNetwork(nn.Module):
    def __init__(self, env, args):
        super(PolicyNetwork, self).__init__()
        assert len(env.action_shape) == 1
        state_components = env.state_shape[0]
        action_components = env.action_shape[0]

        self.nn = nn.Sequential(
            nn.Linear(state_components, args.hidden_layer),
            nn.ReLU(),
            nn.Linear(args.hidden_layer, action_components),
            SigmoidRescaled(ranges=map(torch.tensor, env.action_ranges))
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.learning_rate)

    def train(self, states, critic):
        q_values = -torch.mean(critic(states, self.forward(states)))
        self.optimizer.zero_grad()
        q_values.backward()
        self.optimizer.step()

    def forward(self, states):
        return self.nn(states)

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super(QNetwork, self).__init__()
        assert len(env.action_shape) == 1
        state_components = env.state_shape[0]
        action_components = env.action_shape[0]

        self.value = nn.Sequential(
            nn.Linear(state_components, args.hidden_layer),
            nn.ReLU(),
        )
        self.action_values = nn.Sequential(
            nn.Linear(args.hidden_layer + action_components, 2*args.hidden_layer),
            nn.ReLU(),
            nn.Linear(2*args.hidden_layer, 1),
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=10*args.learning_rate)

    def train(self, states, actions, returns):
        q_values = self.forward(states, actions.view(-1, 1))
        loss = F.mse_loss(returns, q_values).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, states, actions):
        return self.action_values(torch.cat((self.value(states), actions), dim=1))

class Network:
    def __init__(self, env, args, device):
        assert len(env.action_shape) == 1
        # Create `actor` network, starting with `inputs` and returning
        # `action_components` values for each batch example. Usually, one
        # or two hidden layers are employed. Each `action_component[i]` should
        # be mapped to range `[actions_lows[i]..action_highs[i]]`, for example
        # using `tf.nn.sigmoid` and suitable rescaling.
        self._actor = PolicyNetwork(env, args).to(device)

        # Then, create a target actor as a copy of the model using
        # `tf.keras.models.clone_model`.
        self._target_actor = PolicyNetwork(env, args).to(device)
        self._target_actor.load_state_dict(self._actor.state_dict())

        # Create `critic` network, starting with `inputs` and `actions`
        # and producing a vector of predicted returns. Usually, `inputs` are fed
        # through a hidden layer first, and then concatenated with `actions` and fed
        # through two more hidden layers, before computing the returns.
        self._critic = QNetwork(env, args).to(device)

        # Then, create a target critic as a copy of the model using `tf.keras.models.clone_model`.
        self._target_critic = QNetwork(env, args).to(device)
        self._target_critic.load_state_dict(self._critic.state_dict())

        self.tau = args.target_tau
        self.gamma = args.gamma
        if args.recodex:
            self._actor.load_state_dict(torch.load("model_ddpg_actor.py", map_location=device))
            self._target_actor.load_state_dict(torch.load("model_ddpg_actor.py", map_location=device))
            self._critic.load_state_dict(torch.load("model_ddpg_critic.py", map_location=device))
            self._target_critic.load_state_dict(torch.load("model_ddpg_critic.py", map_location=device))

    def _expected_target_returns(self, states, rewards, dones, next_states):
        actions = self._target_actor(next_states)
        q_next_values = self._target_critic(next_states, actions)
        return (rewards.view(-1, 1) + self.gamma * q_next_values.view(-1, 1) * ~dones.view(-1, 1)).detach()

    def _update_target(self, model, target):
        params1 = model.named_parameters()
        params2 = target.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                param2 = dict_params2[name1]
                dict_params2[name1].data.copy_(self.tau * param1.data + (1 - self.tau) * param2.data)
        target.load_state_dict(dict_params2)

    def train(self, states, actions, rewards, dones, next_states):
        # Train separately the actor and critic.
        #
        # Furthermore, update the weights of the target actor and critic networks
        # by using args.target_tau option.
        returns = self._expected_target_returns(states, rewards, dones, next_states)
        self._actor.train(states, self._critic)
        self._critic.train(states, actions, returns)
        self._update_target(self._actor, self._target_actor)
        self._update_target(self._critic, self._target_critic)
        torch.save(self._actor.state_dict(), "model_ddpg_actor.py")
        torch.save(self._critic.state_dict(), "model_ddpg_critic.py")

    def predict_actions(self, states):
        # Compute actions by the actor
        return self._actor(states).detach()

    def predict_values(self, states):
        # Predict actions by the target actor and evaluate them using
        # target_critic.
        actions = self._target_actor(states)
        return self._target_critic(states, actions).detach()

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, shape, mu, theta, sigma, device):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.device = device
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return torch.tensor([self.state], device=self.device)


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=128, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=0.001, type=float, help="Target network update weight.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--goal", default=-130, type=int, help="Goal value for task.")
    parser.add_argument("--recodex", default=True, type=int, help="Running in ReCodEx?")

    args = parser.parse_args()

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    network = Network(env, args, device)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    noise = OrnsteinUhlenbeckNoise(env.action_shape[0], 0., args.noise_theta, args.noise_sigma, device)
    while not args.recodex:
        # Training
        for _ in range(args.evaluate_each):
            state, done, episode_return = env.reset(), False, 0
            noise.reset()
            while not done:
                # Perform an action and store the transition in the replay buffer
                action = network.predict_actions(torch.tensor([state], device=device, dtype=torch.float32)) + noise.sample()
                next_state, reward, done, _ = env.step(action.numpy()[0])

                replay_buffer.append(Transition(state, action, reward, done, next_state))
                # If the replay_buffer is large enough, perform training
                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(lambda p: torch.tensor(p, device=device, dtype=torch.float32), zip(*[replay_buffer[i] for i in batch]))
                    # Perform the training
                    network.train(states, actions, rewards, dones.bool(), next_states)
                state = next_state

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action = network.predict_actions(torch.tensor([state], device=device, dtype=torch.float32))
                state, reward, done, _ = env.step(action.numpy()[0])
                returns[-1] += reward
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))
        if np.mean(returns) >= args.goal:
            break

    # On the end perform final evaluations with `env.reset(True)`
        # On the end perform final evaluations with `env.reset(True)`
    while True:
        state, done = env.reset(True), False
        while not done:
            action = network.predict_actions(torch.tensor([state], device=device, dtype=torch.float32))
            state, reward, done, _ = env.step(action.numpy()[0])
