import argparse # pip3 install argparse
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
plt.rcParams["figure.figsize"] = (15, 7)
plt.style.use('dark_background')

import torch.utils.data

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=256, type=int)
parser.add_argument('-chars_include', default='ABC', type=str)
parser.add_argument('-samples_per_class', default=1000, type=int)

# parser.add_argument('-learning_rate', default=3e-4, type=float)
parser.add_argument('-learning_rate_g', default=3e-4, type=float)
parser.add_argument('-learning_rate_d', default=1e-4, type=float)


parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-discriminator_n', default=1, type=int)

parser.add_argument('-coef_alpha', default=50, type=float)
parser.add_argument('-coef_beta', default=50, type=float)

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

COEF_ALPHA = args.coef_alpha
COEF_BETA = args.coef_beta

DISCRIMINATOR_N = args.discriminator_n

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

class ModelE(torch.nn.Module):
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
            torch.nn.Linear(in_features=64, out_features=Z_SIZE),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x_enc = self.encoder.forward(x)
        x_enc_flat = x_enc.squeeze() # B, 64, 1, 1, => B, 64
        y_prim = self.mlp.forward(x_enc_flat)
        return y_prim


class ModelD(torch.nn.Module): #Discriminator
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
            torch.nn.Linear(in_features=64, out_features=3), # 0 - fake_source, 1 -fake_target, 2-real_target
            torch.nn.Softmax(dim=-1)

        )

    def forward(self, x):
        x_enc = self.encoder.forward(x)
        x_enc_flat = x_enc.squeeze() # B, 64, 1, 1, => B, 64
        y_prim = self.mlp.forward(x_enc_flat)
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

    def forward(self, z):
        z_flat = self.mlp.forward(z)
        z_2d = z_flat.view(z.size(0), 128, self.decoder_size, self.decoder_size)
        y_prim = self.decoder.forward(z_2d)
        return y_prim

model_E = ModelE().to(DEVICE)
model_D = ModelD().to(DEVICE)
model_G = ModelG().to(DEVICE)

def get_param_count(model):
    params = list(model.parameters())
    result = 0
    for param in params:
        count_param = np.prod(param.size()) # size[0] * size[1] ...
        result += count_param
    return result

print(f'model_D params: {get_param_count(model_D)}')
print(f'model_G params: {get_param_count(model_G)}')
print(f'model_E params: {get_param_count(model_E)}')

optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.learning_rate_d)
optimizer_G = torch.optim.Adam(list(model_G.parameters()) + list(model_E.parameters()), lr=args.learning_rate_g)

metrics = {}
for stage in ['train']:
    for metric in ['loss', 'loss_g', 'loss_d', 'loss_gang', 'loss_const', 'loss_tid']:
        metrics[f'{stage}_{metric}'] = []


for epoch in range(1, 500):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    iter_data_loader_target = iter(data_loader_target)
    n = 0
    for x_s, label_s in tqdm(data_loader_source, desc=stage):
        x_s = x_s.to(DEVICE)
        x_t, label_t = next(iter_data_loader_target)
        x_t = x_t.to(DEVICE)
        if n == 0:
            z_s = model_E.forward(x_s)
            z_t = model_E.forward(x_t)

            g_s = model_G.forward(z_s)
            g_t = model_G.forward(z_t)

            z_z_s = model_E.forward(g_s) # Maybe remove detach()

            for p in model_D.parameters():
                p.requires_grad = False

            y_g_s = model_D.forward(g_s)
            y_g_t = model_D.forward(g_t)

            # 0 - fake_source, 1 -fake_target, 2-real_target
            loss_gang = -torch.mean(torch.log(y_g_s[:, 2] + 1e-8)) - torch.mean(torch.log(y_g_t[:, 2] + 1e-8))
            loss_const = torch.mean((z_s-z_z_s)**2)
            loss_tid = torch.mean((x_t-g_t)**2)
            loss_g = loss_gang + COEF_ALPHA*loss_const + COEF_BETA*loss_tid
            loss_g.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()
        else:
            z_s = model_E.forward(x_s)
            z_t = model_E.forward(x_t)

            g_s = model_G.forward(z_s)
            g_t = model_G.forward(z_t)
            for p in model_D.parameters():
                p.requires_grad = True

            y_g_s = model_D.forward(g_s.detach())
            y_g_t = model_D.forward(g_t.detach())
            y_t = model_D.forward(x_t)

            # 0 - fake_source, 1 -fake_target, 2-real_target
            loss_d = - torch.mean(torch.log(y_g_s[:, 0] + 1e-8)) \
                     - torch.mean(torch.log(y_g_t[:, 1] + 1e-8)) \
                     - torch.mean(torch.log(y_t[:, 2] + 1e-8))

            loss_d.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()

            if n >= DISCRIMINATOR_N:
                n = -1
                loss = loss_g + loss_d
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
                metrics_epoch[f'{stage}_loss_gang'].append(loss_gang.cpu().item())
                metrics_epoch[f'{stage}_loss_const'].append(loss_const.cpu().item())
                metrics_epoch[f'{stage}_loss_tid'].append(loss_tid.cpu().item())
                metrics_epoch[f'{stage}_loss_g'].append(loss_g.cpu().item())
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

    plt.subplot(222)  # row col idx
    grid_img = torchvision.utils.make_grid(
        x_s.detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.subplot(224)  # row col idx
    grid_img = torchvision.utils.make_grid(
        g_s.detach().cpu(),
        padding=10,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.subplot(221)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    plt.tight_layout(pad=0.5)

    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss'][-1]):
            exit()
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')


