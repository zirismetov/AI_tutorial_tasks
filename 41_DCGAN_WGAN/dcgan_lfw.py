import argparse # pip3 install argparse
from copy import copy

from torch.utils.data import WeightedRandomSampler
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
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-classes_count', default=10000, type=int)
parser.add_argument('-samples_per_class', default=10000, type=int)

parser.add_argument('-learning_rate', default=3e-4, type=float)
parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import PIL.ImageOps
RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count # 0 = include all
IS_DEBUG = args.is_debug
INPUT_SIZE = 56

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 300 # per class for debugging
    MAX_CLASSES = 6 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 66

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = fetch_lfw_people(resize=None,funneled=True, color=False, min_faces_per_person=50)
        self.n_classes = self.data.target_names.size
        self.labels = self.data.target_names.tolist()
        # self.X = ((np_x - np.mean(np_x, axis=0)) / np.std(np_x, axis=0)).astype(np.float32)
        # self.y = ((np_y - np.mean(np_y, axis=0)) / np.std(np_y, axis=0)).astype(np.float32)

    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        np_x = self.data.images[idx]

        # np_x = np.expand_dims(np_x, axis=0)
        img = Image.fromarray((np_x*255).astype(np.uint8))
        img = img.resize(size=(INPUT_SIZE, INPUT_SIZE))
        # plt.imshow(img, interpolation='nearest')
        # plt.show()
        x = torch.FloatTensor(np.array(img))
        # x = torch.movedim(x, 2, 0)

        np_y = np.zeros((self.n_classes,))
        np_y[self.data.target[idx]] = 1
        y = torch.FloatTensor(np_y)

        return torch.unsqueeze(x, dim=0), self.data.target[idx]


class ModelD(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4,stride=2, padding=1),
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            torch.nn.AdaptiveMaxPool2d(output_size=(1,1))
        )
        self.mpl = torch.nn.Sequential(

            torch.nn.Linear(in_features=64, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_enc = self.encoder.forward(x)
        x_enc_flat = x_enc.squeeze()
        y_prim = self.mpl.forward(x_enc_flat)
        # y_prim = torch.rand(size=(x.size(0), 1))
        return y_prim


class ModelG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_size = INPUT_SIZE // 4
        self.mpl = torch.nn.Linear(Z_SIZE, self.decoder_size**2 * 128)
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),

            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),

            # torch.nn.Upsample(scale_factor=2),
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
        z_flat = self.mpl.forward(z)
        z_2d = z_flat.view(z.size(0), 128, self.decoder_size, self.decoder_size)
        y_prim = self.decoder.forward(z_2d)
        return y_prim


dataset_full = DatasetEMNIST()

np.random.seed(2)
labels_train = copy(dataset_full.labels)
random.shuffle(labels_train, random=np.random.random)
labels_train = labels_train[:MAX_CLASSES]
np.random.seed(int(time.time()))

idx_train = []
str_args_for_hasing = [str(it) for it in [MAX_LEN, MAX_CLASSES] + labels_train]
hash_args = hashlib.md5((''.join(str_args_for_hasing)).encode()).hexdigest()
path_cache = f'../data/{hash_args}_gan.pkl'
if os.path.exists(path_cache):
    print('loading from cache')
    with open(path_cache, 'rb') as fp:
        idx_train = pickle.load(fp)

else:
    labels_count = dict((key, 0) for key in dataset_full.labels)
    for idx, (x, y_idx) in tqdm(
            enumerate(dataset_full),
            'splitting dataset',
            total=len(dataset_full)
    ):
        label = dataset_full.labels[y_idx]
        if MAX_LEN > 0:
            if labels_count[label] >= MAX_LEN:
                if all(it >= MAX_LEN for it in labels_count.values()):
                    break
                continue
        labels_count[label] += 1
        if label in labels_train:
            idx_train.append(idx)

    # with open(path_cache, 'wb') as fp:
    #     pickle.dump(idx_train, fp)

dataset_train = torch.utils.data.Subset(dataset_full, idx_train)
counts = np.bincount(dataset_train.dataset.data.target[dataset_train.indices])
labels_weights = 1. / counts
weights = labels_weights[dataset_train.dataset.data.target[dataset_train.indices]]
sampler = WeightedRandomSampler(weights, len(weights))

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE < 12),
    num_workers=(8 if not IS_DEBUG else 0),
    sampler=sampler
)

model_D = ModelD().to(DEVICE)
model_G = ModelG().to(DEVICE)
def get_parameter_count(model):
    params = list(model.parameters())
    result = 0
    for param in params:
        count_param = np.prod(param.size())
        result += count_param
    return result
print(f"Model_D params :{get_parameter_count(model_D)} ")
print(f"Model_G params :{get_parameter_count(model_G)} ")
# exit()

optimizer = torch.optim.Adam(list(model_D.parameters()) + list(model_G.parameters()), lr=LEARNING_RATE)

metrics = {}
for stage in ['train']:
    for metric in ['loss', 'loss_g', 'loss_d']:
        metrics[f'{stage}_{metric}'] = []

dist_z = torch.distributions.Normal(
    loc=0.0,
    scale=1.0
)

for epoch in range(1, 500):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    for x, x_idx in tqdm(data_loader_train, desc=stage):
        x = x.to(DEVICE)

        z = dist_z.sample((x.size(0), Z_SIZE)).to(DEVICE)
        x_gen = model_G.forward(z)
        for param in model_D.parameters():
            param.requires_grad = False
        y_gen = model_D.forward(x_gen)
        loss_G = -torch.mean(torch.log(y_gen + 1e-8))

        z = dist_z.sample((x.size(0), Z_SIZE)).to(DEVICE)
        x_fake = model_G.forward(z)
        for param in model_D.parameters():
            param.requires_grad = True
        y_fake = model_D.forward(x_fake.detach())
        y_real = model_D.forward(x)
        loss_D = -torch.mean(torch.log(y_real+1e-8) + torch.log(1-y_fake + 1e-8))

        loss = loss_D + loss_G

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
        metrics_epoch[f'{stage}_loss_g'].append(loss_G.cpu().item())
        metrics_epoch[f'{stage}_loss_d'].append(loss_D.cpu().item())

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')
    if epoch % 10 == 0:
        plt.clf()
        plt.subplot(121) # row col idx
        plts = []
        c = 0
        for key, value in metrics.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1
        plt.legend(plts, [it.get_label() for it in plts])

        plt.subplot(122)  # row col idx
        grid_img = torchvision.utils.make_grid(
            x_fake.detach().cpu(),
            padding=10,
            scale_each=True,
            nrow=8
        )
        plt.imshow(grid_img.permute(1, 2, 0))

        plt.tight_layout(pad=0.5)

        if len(RUN_PATH) == 0:
            plt.show()
        else:
            if np.isnan(metrics[f'train_loss'][-1]) or np.isinf(metrics[f'train_loss'][-1]):
                exit()
            plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')