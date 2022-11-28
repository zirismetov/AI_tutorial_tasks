import argparse # pip3 install argparse
import itertools
from copy import copy

from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import hashlib
import os
import pickle
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import random
import torch.distributed
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 15)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=1000, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-chars_include', default='ABC', type=str) #DEFGHIJKLMNOPQRSTUVWXYZ
parser.add_argument('-samples_per_class', default=1000, type=int)

parser.add_argument('-learning_rate_g', default=3e-4, type=float)
parser.add_argument('-learning_rate_d', default=1e-4, type=float)

parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-coef_s', default=10, type=float)
parser.add_argument('-coef_t', default=10, type=float)
parser.add_argument('-coef_i', default=0.5, type=float)

parser.add_argument('-is_debug', default=True, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
Z_SIZE = args.z_size
DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
CHARS_INCLUDE = args.chars_include # '' = include all
IS_DEBUG = args.is_debug
INPUT_SIZE = 28

if not torch.cuda.is_available() or IS_DEBUG:
    IS_DEBUG = True
    MAX_LEN = 300 # per class for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 66

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='byclass',
            train=(MAX_LEN == 0),
            download=True
        )
        self.labels = self.data.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.transpose(np.array(pil_x)).astype(np.float32)
        np_x = np.expand_dims(np_x, axis=0) / 255.0 # (1, W, H) => (1, 28, 28)
        return np_x, y_idx


class DatasetAriel(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/ariel_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1636095494-dml-course-2021-q4/ariel_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.expand_dims(np.transpose(self.X[idx]), axis=0).astype(np.float32)
        y = self.Y[idx]
        return x, y


def filter_dataset(dataset, name):
    str_args_for_hasing = [str(it) for it in [MAX_LEN, CHARS_INCLUDE] + dataset.labels]
    hash_args = hashlib.md5((''.join(str_args_for_hasing)).encode()).hexdigest()
    path_cache = f'../data/{hash_args}_dtngan_{name}.pkl'
    if os.path.exists(path_cache):
        print('loading from cache')
        with open(path_cache, 'rb') as fp:
            idxes_include = pickle.load(fp)
    else:
        idxes_include = []
        char_counter = {}
        for char in CHARS_INCLUDE:
            char_counter[char] = 0
        idx = 0
        for x, y in tqdm(dataset, f'filter_dataset {name}'):
            char = dataset.labels[int(y)]
            if char in CHARS_INCLUDE:
                char_counter[char] +=1
                if char_counter[char] < MAX_LEN:
                    idxes_include.append(idx)
            if all(it >= MAX_LEN for it in char_counter.values()):
                break
            idx += 1
        with open(path_cache, 'wb') as fp:
            pickle.dump(idxes_include, fp)

    dataset_filtered = torch.utils.data.Subset(dataset, idxes_include)
    return dataset_filtered


dataset_source = DatasetEMNIST()
dataset_target = DatasetAriel()

dataset_source = filter_dataset(dataset_source, 'dataset_source')
dataset_target = filter_dataset(dataset_target, 'dataset_target')

print(f'dataset_source: {len(dataset_source)} dataset_target: {len(dataset_target)}')

data_loader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_source) % BATCH_SIZE < 12),
    num_workers=(8 if not IS_DEBUG else 0)
)

data_loader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_target) % BATCH_SIZE < 12),
    num_workers=(8 if not IS_DEBUG else 0)
)

class ModelE(torch.nn.Module): # in equations f(x) Encoder
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # B, 4, 14, 14

            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # B, 32, 7, 7

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # B, 64, 4,4

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveMaxPool2d(output_size=(1,1)) # B, 64, 1, 1
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=Z_SIZE)
        )

    def forward(self, x):
        x_enc = self.encoder.forward(x)
        x_enc_flat = x_enc.squeeze() # B, 64, 1, 1, => B, 64
        y_prim = self.mlp.forward(x_enc_flat)
        return y_prim


class ModelD(torch.nn.Module): # Discriminator
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # B, 4, 14, 14

            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # B, 32, 7, 7

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # B, 64, 4,4

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),

            torch.nn.AdaptiveMaxPool2d(output_size=(1,1)) # B, 64, 1, 1
        )

    def forward(self, x):
        x_enc = self.encoder.forward(x)
        y_prim = x_enc.squeeze() # B, 64, 1, 1, => B, 64
        return y_prim


class ModelG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_size = INPUT_SIZE // 4 # upsample twice * 2dim (W, H)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(Z_SIZE, self.decoder_size ** 2 * 128)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),

            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),

            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),

            torch.nn.Sigmoid()
        )
        self.encoder = ModelE()

    def forward(self, x):
        z = self.encoder.forward(x)
        z_flat = self.mlp.forward(z)
        z_2d = z_flat.view(z.size(0), 128, self.decoder_size, self.decoder_size)
        y_prim = self.decoder.forward(z_2d) # (B, 1, 28, 28)
        return y_prim

model_G_s_t = ModelG().to(DEVICE)
model_G_t_s = ModelG().to(DEVICE)

model_D_s = ModelD().to(DEVICE)
model_D_t = ModelD().to(DEVICE)

params_G = itertools.chain(model_G_s_t.parameters(), model_G_t_s.parameters())
params_D = itertools.chain(model_D_s.parameters(), model_D_t.parameters())

opt_G = torch.optim.Adam(params_G, lr=args.learning_rate_g)
opt_D = torch.optim.Adam(params_D, lr=args.learning_rate_d)

metrics = {}
for stage in ['train']:
    for metric in [
        'loss_g',
        'loss_d',
        'loss_c',
        'loss_i'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    iter_data_loader_target = iter(data_loader_target)
    x_s_prev = None
    n = 0
    for x_s, label_s in tqdm(data_loader_source, desc=stage):
        x_t, label_t = next(iter_data_loader_target)

        if x_s.size() != x_t.size():
            break

        x_s = x_s.to(DEVICE)
        x_t = x_t.to(DEVICE)
        x_s_prev = x_s

        opt_G.zero_grad()
        opt_D.zero_grad()
        if n % 2 == 0:
            for param in params_D:
                param.requires_grad = False
            g_t = model_G_s_t.forward(x_s)
            g_s = model_G_t_s.forward(g_t)

            y_g_t = model_D_t.forward(g_t) # 1 - real, 0 - fake
            y_g_s = model_D_s.forward(g_s) # 1 - real, 0 - fake

            loss_g_t = torch.mean(torch.abs(y_g_t- 1.0))
            loss_g_s = torch.mean(torch.abs(y_g_s - 1.0))
            loss_g = loss_g_t + loss_g_s
            loss_c_s = torch.mean(torch.abs(g_s - x_s)) * args.coef_s

            g_s_prim = model_G_t_s.forward(x_t)
            g_t_prim = model_G_s_t.forward(g_s_prim)
            loss_c_t = torch.mean(torch.abs(g_t_prim - x_t)) * args.coef_t

            y_g_t_prim = torch.mean(torch.abs(g_t_prim - 1.0))
            y_g_s_prim = torch.mean(torch.abs(g_s_prim - 1.0))
            loss_g_t_prim = torch.mean(torch.abs(y_g_t_prim - 1.0))
            loss_g_s_prim = torch.mean(torch.abs(y_g_s_prim - 1.0))
            loss_g += loss_g_t_prim + loss_g_s_prim

            loss_c = loss_c_t + loss_c_s

            i_t = model_G_s_t.forward(x_t)
            i_s = model_G_t_s.forward(x_s)
            loss_i_t = torch.mean(torch.abs(i_t - x_t)) * args.coef_i
            loss_i_s = torch.mean(torch.abs(i_s - x_s)) * args.coef_i
            loss_i = loss_i_t + loss_i_s

            loss = loss_c + loss_g + loss_i
            loss.backward()
            opt_G.step()

            metrics_epoch[f'{stage}_loss_i'].append(loss_i.cpu().item())
            metrics_epoch[f'{stage}_loss_g'].append(loss_g.cpu().item())
            metrics_epoch[f'{stage}_loss_c'].append(loss_c.cpu().item())
        else:
            for param in params_D:
                param.requires_grad = True
            g_t = model_G_s_t.forward(x_s)
            g_s = model_G_t_s.forward(g_t)

            y_g_t = model_D_t.forward(g_t.detach())  # 1 - real, 0 - fake
            y_g_s = model_D_s.forward(g_s.detach())  # 1 - real, 0 - fake

            y_x_t = model_D_t.forward(x_t)  # 1 - real, 0 - fake
            y_x_s = model_D_s.forward(x_s)

            loss_g_t = torch.mean(y_g_t**2)
            loss_g_s = torch.mean(y_g_s ** 2)
            loss_x_t = torch.mean((y_x_t - 1.0) ** 2)
            loss_x_s = torch.mean((y_x_s - 1.0) ** 2)


            loss_d = loss_g_t + loss_g_s + loss_x_t + loss_x_s
            loss_d.backward()
            opt_D.step()

            metrics_epoch[f'{stage}_loss_d'].append(loss_d.cpu().item())
        n += 1


    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.cla()

    plt.subplot(3, 1, 1)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    plt.subplot(3, 1, 2)  # row col idx
    grid_img = torchvision.utils.make_grid(
        x_s_prev.detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.subplot(3, 1, 3)  # row col idx
    grid_img = torchvision.utils.make_grid(
        g_t.detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.tight_layout(pad=0.5)

    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss_g'][-1]) or np.isnan(metrics[f'train_loss_d'][-1]):
            exit()
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')


