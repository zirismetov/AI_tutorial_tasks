import os
import gym #  pip install git+https://github.com/openai/gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import torch.distributions
# !pip3 install box2d-py
import glob
import io
import base64
# from gym.wrappers import Monitor
# from IPython.display import HTML
# from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
import time
import matplotlib as mpl
import pdb
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.distributions import Categorical
# !pip3 install box2d-py

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('animation', html='jshtml')
display = Display(visible=0, size=(1400, 900)).start()

# conda install box2d-py
# export HEADLESS=1

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-is_render', default=('HEADLESS' not in os.environ), type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-replay_buffer_size', default=20000, type=int)

parser.add_argument('-hidden_size', default=512, type=int)

parser.add_argument('-gamma', default=0.8, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=200, type=int)
parser.add_argument('-target_update', default=500, type=int)

args, other_args = parser.parse_known_args()

import matplotlib
if args.is_render:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')

if not torch.cuda.is_available():
    args.device = 'cpu'

RUN_PATH = ''
if not args.is_render:
    RUN_PATH = f'ppo_{int(time.time())}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)


class ModelActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=action_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, s_t0):
        return self.layers.forward(s_t0)


class ModelCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=1),
        )

    def forward(self, s_t0):
        return self.layers.forward(s_t0)

class ReplayPriorityMemory:
    def __init__(self, size, batch_size, prob_alpha=1):
        self.size = size
        self.batch_size = batch_size
        self.prob_alpha = prob_alpha
        self.memory = []
        self.Rs = []
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        new_priority = np.mean(self.priorities) if self.memory else 1.0

        self.memory.append(transition)
        self.Rs.append(transition[-1])
        if len(self.memory) > self.size:
            del self.memory[0]
            del self.Rs[0]
        pos = len(self.memory) - 1
        self.priorities[pos] = new_priority


    def sample(self):
        probs = np.array(self.priorities)
        if len(self.memory) < len(probs):
            probs = probs[:len(self.memory)]

        probs = probs - np.min(probs)

        probs += 1e-8
        probs = probs ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), args.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority.item()

    def __len__(self):
        return len(self.memory)


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.is_double = True

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = args.gamma    # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.device = args.device

        self.model_a = ModelActor(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.model_t = ModelActor(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.policy_clip = 0.2

        self.model_c = ModelCritic(self.state_size, self.action_size, args.hidden_size).to(self.device)

        self.optimizer_a = torch.optim.Adam(
            self.model_a.parameters(),
            lr=self.learning_rate,
        )
        self.optimizer_c = torch.optim.Adam(
            self.model_c.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = ReplayPriorityMemory(args.replay_buffer_size, args.batch_size)
        self.update_model_t()

    def update_model_t(self):
        self.model_t.load_state_dict(self.model_a.state_dict())
        for param in self.model_t.parameters():
            param.requires_grad = False

    def act(self, s_t0):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                s_t0 = torch.FloatTensor(s_t0).to(args.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                self.model_a = self.model_a.eval()
                q_all = self.model_a.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self, e):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch, batch_idxes = self.replay_memory.sample()
        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()

        s_t0, a_t0, delta = zip(*batch)
        s_t0 = torch.FloatTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)

        delta = torch.FloatTensor(delta).to(args.device)

        v_t = self.model_c.forward(s_t0).squeeze()

        A = delta - v_t.detach()
        A = (A - A.mean()) / (A.std() + 1e-8)
        loss_c = (delta - v_t)**2
        loss_c_mean = torch.mean(loss_c)
        loss_c_mean.backward()
        self.optimizer_c.step()

        self.model_a = self.model_a.train()

        a_all = self.model_a.forward(s_t0)
        a_t = a_all[range(len(a_t0)), a_t0]
        #         a_t = a_t.argmax(dim=1)
        #         set_trace()
        #         dist = Categorical(a_all)
        #         a_theta = dist.log_prob(a_t)

        a_all_old = self.model_t.forward(s_t0).detach()
        a_t_old = a_all_old[range(len(a_t0)), a_t0]
        #         a_t_old = a_t_old.argmax(dim=1)
        #         dist = Categorical(a_all_old)
        if e < args.target_update + 1:
            a_t_old = a_t.clone().detach()
        #         a_old_theta = dist.log_prob(a_t_old)

        ratios = torch.exp(a_t - a_t_old)
        clamp = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * A
        final = torch.min(ratios, clamp)
        loss_a = -(A * final)
        loss_a_mean = torch.mean(loss_a)
        loss_a_mean.backward()
        self.optimizer_a.step()

        self.replay_memory.update_priorities(batch_idxes, loss_a)

        return loss_a_mean.item(), loss_c_mean.item()

# environment name
env = gym.make('LunarLander-v2')
plt.figure()

all_scores = []
all_losses = []
all_losses_a = []
all_losses_c = []
all_t = []

agent = A2CAgent(
    env.observation_space.shape[0], # first 2 are position in x axis and y axis(hieght) , other 2 are the x,y axis velocity terms, lander angle and angular velocity, left and right left contact points (bool)
    env.action_space.n
)
is_end = False
PLOT_REFRESH_RATE = 10