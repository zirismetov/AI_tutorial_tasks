import os

import gym #  pip install git+https://github.com/openai/gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import torch.distributions
# !pip3 install box2d-py

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-is_render', default=('COLAB_GPU' not in os.environ), type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=512, type=int)
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-replay_buffer_size', default=5000, type=int)

parser.add_argument('-hidden_size', default=128, type=int)
parser.add_argument('-target_update', default=3000, type=int)

parser.add_argument('-gamma', default=0.9, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=500, type=int)

args, other_args = parser.parse_known_args()

import matplotlib
if args.is_render:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    args.device = 'cpu'

class Model(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Model, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.LayerNorm(normalized_shape=hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_size, out_features=action_size),
            torch.nn.Softmax(dim=1)
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
        new_priority = np.max(self.priorities) if self.memory else 1.0

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


class PGAgent:
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
        self.p_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.t_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(
            self.p_model.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = ReplayPriorityMemory(args.replay_buffer_size, args.batch_size)
    def update_target_model(self):
        self.t_model.load_state_dict(self.p_model.state_dict())

    def act(self, s_t0):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                s_t0 = torch.FloatTensor(s_t0).to(args.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                q_all = self.p_model.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch, batch_idxes = self.replay_memory.sample()
        self.optimizer.zero_grad()

        s_t0, a_t0, r_c = zip(*batch)
        idxes = torch.arange(len(s_t0)).to(args.device)

        s_t0 = torch.FloatTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)
        r_c = torch.FloatTensor(r_c).to(args.device)

        a_all = self.p_model.forward(s_t0)
        dist = torch.distributions.Categorical(a_all)
        a_t = dist.log_prob(a_t0)

        a_all = self.t_model.forward(s_t0).detach()
        dist = torch.distributions.Categorical(a_all)
        a_target = dist.log_prob(a_t0)

        r_c = (r_c - np.mean(self.replay_memory.Rs)) / np.std(self.replay_memory.Rs)
        loss = -r_c * torch.log((a_t + 1e-8) / (a_target + 1e-8))

        self.replay_memory.update_priorities(batch_idxes, loss)
        loss = torch.mean(loss)

        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

# environment name
env = gym.make('LunarLander-v2')
plt.figure()

all_scores = []
all_losses = []
all_t = []

agent = PGAgent(
    env.observation_space.shape[0], # first 2 are position in x axis and y axis(hieght) , other 2 are the x,y axis velocity terms, lander angle and angular velocity, left and right left contact points (bool)
    env.action_space.n
)
is_end = False

R_min = float('Inf')
frames = 0
for e in range(args.episodes):
    s_t0 = env.reset()
    reward_total = 0

    transitions = []
    for t in range(args.max_steps):
        if args.is_render and len(all_scores):
            if all_scores[-1] > 0:
                env.render()
        a_t0 = agent.act(s_t0)
        s_t1, r_t1, is_end, _ = env.step(a_t0)

        reward_total += r_t1

        if t == args.max_steps-1:
            r_t1 = -100
            is_end = True

        transitions.append([s_t0, a_t0, r_t1])
        s_t0 = s_t1

        if is_end:
            all_scores.append(reward_total)
            break

    for t, transition in enumerate(transitions):
        s_t0, a_t0, r_t1 = transition
        R_t = 0

        for t_future, transition_f in enumerate(transitions[t:]):
            _, _, r_t1_f = transition_f
            R_t += (args.gamma ** t_future) * r_t1_f
        agent.replay_memory.push((s_t0, a_t0, R_t))

    e_losses = []
    for _ in range(int(1.5 * np.ceil(len(transitions) / args.batch_size))):
        if len(agent.replay_memory) > args.batch_size:
            loss = agent.replay()
            e_losses.append(loss)
    all_losses.append(np.mean(e_losses))

    frames += len(transitions)
    if frames > args.target_update:
        agent.update_target_model()
        frames = 0

    all_t.append(t)
    print(
        f'episode: {e}/{args.episodes} '
        f'loss: {all_losses[-1] if len(all_losses) else 0} '
        f'score: {reward_total} '
        f't: {t} '
        f'e: {agent.epsilon}')

    if e % 100 == 0:
        plt.clf()

        plt.subplot(3, 1, 1)
        plt.ylabel('Score')
        plt.plot(all_scores)

        plt.subplot(3, 1, 2)
        plt.ylabel('Loss')
        plt.plot(all_losses)

        plt.subplot(3, 1, 3)
        plt.ylabel('Steps')
        plt.plot(all_t)

        plt.xlabel('Episode')
        plt.pause(1e-3)  # pause a bit so that plots are updated

env.close()
plt.ioff()
plt.show()