import gym #  pip install git+https://github.com/openai/gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import glob
import io
import base64
# from gym.wrappers import Monitor
# from IPython.display import HTML
# from pyvirtualdisplay import Display
# from IPython import display as ipythondisplay
import time
import matplotlib
import pdb
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# !pip3 install box2d-py

# display = Display(visible=0, size=(1400, 900))
# display.start()

"""
Utility functions to enable video recording of gym environment 
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""

# def show_video():
#     mp4list = glob.glob('video/*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         ipythondisplay.display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-is_render', default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-episodes', default=1000, type=int)
parser.add_argument('-replay_buffer_size', default=5000, type=int)
parser.add_argument('-target_update', default=3000, type=int)

parser.add_argument('-hidden_size', default=128, type=int)

parser.add_argument('-gamma', default=0.4, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=5000, type=int)

args, other_args = parser.parse_known_args()

if not torch.cuda.is_available():
    args.device = 'cpu'

class Model(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Model, self).__init__()
        self.input_dim = state_size
        self.output_dim = action_size

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.LayerNorm(normalized_shape=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

        self.values = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size,out_features=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=1),
            torch.nn.LeakyReLU(),

        )
        self.advantages = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=action_size),
            torch.nn.LeakyReLU(),

        )
    def forward(self, s_t0):
        features = self.layers.forward(s_t0)
        values = self.values.forward(features)
        advantages = self.advantages.forward(features)
        qvals = values + (advantages - advantages.mean())
        return qvals


class ReplayPriorityMemory:
    def __init__(self, size, batch_size, prob_alpha=1):
        self.size = size
        self.batch_size = batch_size
        self.prob_alpha = prob_alpha
        self.memory = []
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        new_priority = np.median(self.priorities) if self.memory else 1.0

        self.memory.append(transition)
        if len(self.memory) > self.size:
            del self.memory[0]
        pos = len(self.memory) - 1
        self.priorities[pos] = new_priority

    def sample(self):
        probs = np.array(self.priorities)
        if len(self.memory) < len(probs):
            probs = probs[:len(self.memory)]

        probs += 1e-8
        probs = probs ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority.item()

    def __len__(self):
        return len(self.memory)


class DQNAgent:
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
        self.q_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = ReplayPriorityMemory(args.replay_buffer_size, args.batch_size)

    def act(self, s_t0):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                s_t0 = torch.FloatTensor(s_t0).to(args.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                q_all = self.q_model.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.optimizer.zero_grad()
        batch, replay_idx = self.replay_memory.sample()
        s_t0, a_t0, r_t1, s_t1, is_end = zip(*batch)

        s_t0 = torch.FloatTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)
        r_t1 = torch.FloatTensor(r_t1).to(args.device)
        s_t1 = torch.FloatTensor(s_t1).to(args.device)
        is_not_end = torch.FloatTensor((np.array(is_end) == False) * 1.0).to(args.device)

        idxes = torch.arange(args.batch_size).to(args.device)

        q_t0_all = self.q_model.forward(s_t0)
        q_t0 = q_t0_all[idxes, a_t0]

        q_t1_all = self.q_model.forward(s_t1)
        a_t1 = q_t1_all.argmax(dim=1)
        q_t1 = q_t1_all[idxes, a_t1]

        q_t1_final = r_t1 + is_not_end * (args.gamma * q_t1)

        td_error = (q_t0 - q_t1_final)**2
        self.replay_memory.update_priorities(replay_idx, td_error)
        loss = torch.mean(td_error)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

# environment name
env = gym.make('MountainCar-v0')
plt.figure()

all_scores = []
all_losses = []
all_t = []

agent = DQNAgent(
    env.observation_space.shape[0], # first 2 are position in x axis and y axis(hieght) , other 2 are the x,y axis velocity terms, lander angle and angular velocity, left and right left contact points (bool)
    env.action_space.n
)
is_end = False

# env = Monitor(env, f'./video_MountainCar', video_callable=lambda episode_id: True, force=True)

for e in range(args.episodes+1):
    s_t0 = env.reset()
    reward_total = 0
    episode_loss = []

    for t in range(args.max_steps):
        a_t0 = agent.act(s_t0)

        if e != 0 and e % 100 == 0 or all_scores and all([it > 0 for it in all_scores[-2:]]):
            print(f"THIS ONE: {e}")
            env.render()

        s_t1, r_t1, is_end, _ = env.step(a_t0)
        if t == args.max_steps-1:
            r_t1 = -100
            is_end = True
        reward_total += r_t1
        # exploration alg -> replay memory next embedding and previous (current)
        # curiosity model (dqn)

        agent.replay_memory.push(
            (s_t0, a_t0, r_t1, s_t1, is_end)
        )
        s_t0 = s_t1
        if len(agent.replay_memory) > args.batch_size:
            # pdb.set_trace()
            loss = agent.replay()
            episode_loss.append(loss)

        if is_end:
            all_scores.append(reward_total)
            all_losses.append(np.mean(episode_loss))
            break

    all_t.append(t)
    print(
        f'episode: {e}/{args.episodes} '
        f'loss: {all_losses[-1]} '
        f'score: {reward_total} '
        f't: {t} '
        f'e: {agent.epsilon}')

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
    plt.pause(1e-2)  # pause a bit so that plots are updated

# show_video()
env.close()
plt.ioff()
plt.show()