#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import collections

import car_racing_evaluator

class NoisyLinear(nn.Module):
    def __init__(self, x, y, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.x = x
        self.y = y
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(y, x))
        self.weight_sigma = nn.Parameter(torch.empty(y, x))
        self.register_buffer('weight_epsilon', torch.empty(y, x))
        self.bias_mu = nn.Parameter(torch.empty(y))
        self.bias_sigma = nn.Parameter(torch.empty(y))
        self.register_buffer('bias_epsilon', torch.empty(y))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.x)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.x))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.y))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.x)
        epsilon_out = self._scale_noise(self.y)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)

class WeightedLoss(nn.Module):
    def forward(self, input, target, weights):
        batch_loss = (torch.abs(input - target)<1).float()*(input - target)**2 +\
            (torch.abs(input - target)>=1).float()*(torch.abs(input - target) - 0.5)
        weighted_batch_loss = weights * batch_loss 
        weighted_loss = weighted_batch_loss.sum()
        return weighted_loss, torch.abs(input - target)

class Network(nn.Module):
    def __init__(self, env, args):
        super(Network, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(args.frame_history, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01)
        )
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(args.input_width, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(args.input_height, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        self.fc1 = NoisyLinear(convw * convh * 64, 512, std_init=0.5)
        self.leaky_relu_fc1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = NoisyLinear(512, args.discretized_actions, std_init=0.5)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=args.alpha)
        self.loss = nn.SmoothL1Loss()
        self.wloss = WeightedLoss()
        self.gamma = args.gamma
        self.n_steps = args.n_steps

    def sample_noise(self):
        self.fc1.sample_noise()
        self.fc2.sample_noise()

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        return self.fc2(self.leaky_relu_fc1(self.fc1(out)))

    def replay(self, states, actions, rewards, dones, next_states, target):
        target.eval()
        target.sample_noise()
        with torch.set_grad_enabled(True):
            argmax_actions = torch.argmax(self.forward(next_states), dim=1)
            q_next = self.forward(next_states)
            y = rewards.view(-1, 1) + (self.gamma ** self.n_steps) * q_next.gather(1, argmax_actions.view(-1, 1)) * ~dones.view(-1, 1)
            y = y.detach()
            self.optimizer.zero_grad()
            self.sample_noise()
            q_state = self.forward(states)
            loss = self.loss(q_state.gather(1, actions.view(-1, 1)), y)
            loss.backward()
            self.optimizer.step()

    def prioritized_replay(self, states, actions, rewards, dones, next_states, target, w):
        target.eval()
        target.sample_noise()
        with torch.set_grad_enabled(True):
            argmax_actions = torch.argmax(self.forward(next_states), dim=1)
            q_next = self.forward(next_states)
            y = rewards.view(-1, 1) + (self.gamma ** self.n_steps) * q_next.gather(1, argmax_actions.view(-1, 1)) * ~dones.view(-1, 1)
            y = y.detach()
            self.optimizer.zero_grad()
            self.sample_noise()
            q_state = self.forward(states)
            wloss, batch = self.wloss(q_state.gather(1, actions.view(-1, 1)), y, w)
            wloss.backward()
            self.optimizer.step()
            return batch

    def argmax(self, state):
        with torch.no_grad():
            q_state = self.forward(state)
        return torch.argmax(q_state).item()

def discrete_action_map_to_continuous(action):
    return {
        ("left", "gas"): (-1, 1, 0),
        ("left", "brake"): (-1, 0, 1),
        ("left", "none"): (-1, 0, 0),
        ("right", "gas"): (1, 1, 0),
        ("right", "brake"): (1, 0, 1),
        ("right", "none"): (1, 0, 0),
        ("none", "gas"): (0, 1, 0),
        ("none", "brake"): (0, 0, 1),
        ("none", "none"): (0, 0, 0),
    }[action]

def continuous_action_map_to_discrete(action):
    return {
        (-1, 1, 0): ("left", "gas"),
        (-1, 0, 1): ("left", "brake"),
        (-1, 0, 0): ("left", "none"),
        (1, 1, 0): ("right", "gas"),
        (1, 0, 1): ("right", "brake"),
        (1, 0, 0): ("right", "none"),
        (0, 1, 0): ("none", "gas"),
        (0, 0, 1): ("none", "brake"),
        (0, 0, 0): ("none", "none"),
    }[action]

def scale_luminance(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=4, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=4, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--input_width", default=96, type=int, help="Width of image input.")
    parser.add_argument("--input_height", default=96, type=int, help="Height of image input.")
    parser.add_argument("--discretized_actions", default=9, type=int, help="Amount of discretized actions.")
    parser.add_argument("--max_reward", default=1e3, type=int, help="Amount of discretized actions.")
    parser.add_argument("--alpha", default=0.0000625, type=float, help="Learning rate.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--recodex", default=False, type=int, help="Running in ReCodEx?")
    parser.add_argument("--goal", default=700, type=int, help="Goal value for task.")
    parser.add_argument("--period", default=100, type=int, help="Period to evaluate goal value for task.")
    parser.add_argument("--update_freq", default=1e4, type=int, help="Frequency of updates to target network.")
    parser.add_argument("--maxlen", default=1000000, type=int, help="Maximum size of replay memory buffer.")
    parser.add_argument("--n_steps", default=3, type=int, help="Perform n steps in n-step methods.")
    parser.add_argument("--alpha_is", default=0.7, type=int, help="Alpha hyperparameter for importance sampling (IS).")
    parser.add_argument("--beta_init_is", default=0.5, type=int, help="Beta hyperparameter for importance sampling (IS).")


    args = parser.parse_args()

    # Fix random seeds and number of threads
    # np.random.seed(42)
    # torch.random.manual_seed(42)

    # Create the environment
    env = car_racing_evaluator.environment(args.frame_skip)

    # Construct the network
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    network = Network(env, args).double().to(device)
    if args.recodex:
        network.load_state_dict(torch.load("model_q_network.py", map_location=device))
    target = Network(env, args).double().to(device)
    target.load_state_dict(network.state_dict())

    replay_buffer = collections.deque(maxlen=args.maxlen)
    replay_errors = collections.deque(maxlen=args.maxlen)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    # Implement a variation to Deep Q Network algorithm.
    action_space = [["left", "right", "none"], ["gas", "brake", "none"]]
    discrete_actions = list(itertools.product(*action_space))
    # action_space = [[-1,0,1], [0,1], [0,1]]
    # continuous_actions = list(itertools.product(*action_space))

    evaluating = args.recodex
    epsilon = args.epsilon
    episode_returns, count = [], 0
    episodes_count = 0
    while not evaluating:
        # Perform episode
        state, done, episode_return = env.reset(), False, 0
        composite_state = args.frame_history * [scale_luminance(state)]
        
        # n-step
        network.sample_noise()
        ai = network.argmax(torch.tensor([composite_state], device=device))
        S, A, R = [composite_state], [ai], [None]
        tau, t, T = 0, 0, float("inf")
        while not tau == T - 1:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()
            if t < T:
                next_state, reward, done, _ = env.step(discrete_action_map_to_continuous(discrete_actions[A[t]]))
                composite_next_state = composite_state[1:args.frame_history] + [scale_luminance(next_state)]
                S += [composite_next_state]
                R += [reward]
                if done:
                    T = t + 1
                else:
                    network.sample_noise()
                    A += [network.argmax(torch.tensor([composite_state], device=device))]
                episode_return += reward
                count += 1
            tau = t - args.n_steps + 1
            if tau >= 0:
                min_i, max_i = tau + 1, min(tau + args.n_steps, T)
                G = np.array([args.gamma ** (i - tau - 1) * R[i] for i in range(min_i, max_i)]).sum()
                if tau + args.n_steps < T:
                    last = tau + args.n_steps + 1 >= T
                    # Append state, action, reward, done and next_state to replay_buffer
                    replay_buffer.append(Transition(S[tau], A[tau + args.n_steps - 1], G, last, S[tau + args.n_steps]))
                    replay_errors.append(args.max_reward)
                    # If the replay_buffer is large enough, preform a training batch
                    # of `args.batch_size` uniformly randomly chosen transitions.
                    dist = np.array(replay_errors)**args.alpha_is
                    dist /= dist.sum()
                    idx = np.random.choice(a=len(replay_errors), size=args.batch_size, p=dist)
                    wIs = (len(replay_buffer) * dist)**(-((1 - args.beta_init_is)*max(1, episodes_count/args.episodes) + args.beta_init_is))
                    wIs /= wIs.max()
                    w = wIs[idx]
                    trans = [replay_buffer[i] for i in idx]
                    states, actions, rewards, dones, next_states = map(lambda p: torch.tensor(p, device=device), zip(*trans))
                    # network.replay(states, actions, rewards, dones, next_states, target)
                    abs_err = network.prioritized_replay(states, actions, rewards, dones, next_states, target, torch.tensor(w, device=device))
                    for i in range(len(abs_err)):
                        replay_errors[ idx[i] ] = abs_err[i].item() + 1e-6
            t += 1
            composite_state = composite_next_state
            if count % args.update_freq == 0:
                target.load_state_dict(network.state_dict())
                # input("TYPE SOMETHING")

        # Decide if we want to start evaluating
        episodes_count += 1
        print("Episode #{}: {}".format(episodes_count, episode_return))
        episode_returns = episode_returns[-(args.period - 1):] + [episode_return]
        evaluating |= np.mean(episode_returns) > args.goal
        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        # else:
        torch.save(network.state_dict(), "model_q_network.py")

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        composite_state = args.frame_history * [state]
        while not done:
            # if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                # env.render()
            composite_state += [state]
            composite_state = args.frame_history * [scale_luminance(state)]
            with torch.no_grad():
                q_state = network(torch.tensor([composite_state]))
            ai = torch.argmax(q_state).item()
            action = discrete_action_map_to_continuous(discrete_actions[ai])
            state, reward, done, _ = env.step(action)


