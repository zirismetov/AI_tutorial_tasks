import argparse  # pip3 install argparse
import hashlib
import os
import pickle
import random
import time
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from scipy.spatial.distance import cdist
from tqdm import tqdm  # pip install tqdm

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import sklearn.decomposition
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from IPython import embed
from sklearn.cluster import KMeans
from numpy import linalg as LA
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=66, type=int)
parser.add_argument('-classes_count', default=20, type=int)
parser.add_argument('-samples_per_class', default=600, type=int)

parser.add_argument('-learning_rate', default=1e-4, type=float)
from torch.autograd import Variable
parser.add_argument('-z_size', default=32, type=int)
parser.add_argument('-margin', default=0.2, type=float)
parser.add_argument('-K', default=8, type=int)
parser.add_argument('-M', default=8, type=int)
parser.add_argument('-D', default=8, type=int)

parser.add_argument(
    '-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
Z_SIZE = args.z_size
MARGIN = args.margin
TRAIN_TEST_SPLIT = 0.8
args.batch_size = args.M * args.D
CLASS_RATIO = 1.5

DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count  # 0 = include all
IS_DEBUG = args.is_debug

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 300  # per class for debugging
    MAX_CLASSES = 6  # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 64


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()  # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='byclass',
            train=(MAX_LEN == 0),
            download=True
        )
        class_to_idx = self.data.class_to_idx
        idx_to_class = dict((value, key) for key, value in class_to_idx.items())
        self.labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.transpose(np.array(pil_x)).astype(np.float32)
        np_x = np.expand_dims(np_x, axis=0)  # (1, W, H) => (1, 28, 28)
        return np_x, y_idx


dataset_full = DatasetEMNIST()

labels_train, labels_test = train_test_split(
    dataset_full.labels[:MAX_CLASSES],
    train_size=TRAIN_TEST_SPLIT,
    shuffle=True,
    random_state=1
)
labels_count = dict((key, 0) for key in dataset_full.labels)

labels_y_train = [dataset_full.labels.index(it) for it in labels_train]
labels_y_test = [dataset_full.labels.index(it) for it in labels_test]

y_to_relative_y_train = torch.LongTensor(labels_y_train).to(DEVICE)
y_to_relative_y_test = torch.LongTensor(labels_y_test).to(DEVICE)

print(
    f'labels_train: {labels_train} '
    f'labels_test: {labels_test} '
)

idx_train = []
idx_test = []
str_args = [str(it) for it in [MAX_LEN, MAX_CLASSES]]
hash_args = hashlib.md5((''.join(str_args)).encode()).hexdigest()
path_cache = f'../data/{hash_args}.pkl'
if os.path.exists(path_cache):
    print('loading from cache')
    with open(path_cache, 'rb') as fp:
        idx_train, idx_test = pickle.load(fp)

else:
    for idx, (x, y_idx) in tqdm(enumerate(dataset_full), 'splitting dataset', total=len(dataset_full)):
        label = dataset_full.labels[y_idx]
        if MAX_LEN > 0:  # for debugging
            if labels_count[label] >= MAX_LEN:
                if all(it >= MAX_LEN for it in labels_count.values()):
                    break
                continue
        labels_count[label] += 1
        if label in labels_train:
            idx_train.append(idx)
        elif label in labels_test:
            idx_test.append(idx)

    with open(path_cache, 'wb') as fp:
        pickle.dump((idx_train, idx_test), fp)

dataset_train = torch.utils.data.Subset(dataset_full, idx_train)
dataset_test = torch.utils.data.Subset(dataset_full, idx_test)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
    # labels_y=labels_y_train
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True
    # labels_y=labels_y_test
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet18(  # 3, W, H
            pretrained=False
        )
        self.encoder.fc = torch.nn.Linear(
            in_features=self.encoder.fc.in_features,
            out_features=Z_SIZE
        )

    def forward(self, x):
        z_raw = self.encoder.forward(x.expand(x.size(0), 3, 28, 28))
        z = F.normalize(z_raw, p=2, dim=-1)
        return z


class Magnet_loss(torch.nn.Module):
    def __init__(self):
        super(Magnet_loss, self).__init__()


    def forward(self, z, y_idx):
        losses = 0
        return losses

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = Magnet_loss().to(DEVICE)

model = model.to(DEVICE)
metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'x', 'z', 'y', 'acc']:
        metrics[f'{stage}_{metric}'] = []

# collect cdist
model = model.eval()
list_of_embs_labels = []
K_cluster_centrs = []
Dist_all = []
init = True
with torch.no_grad():
    # collect emb and k_clusters
    for x, y_idx in tqdm(data_loader_train, desc=stage):
        x = x.to(DEVICE)
        y_idx = y_idx.squeeze().to(DEVICE)
        z = model.forward(x)
        list_of_embs_labels.append({'emb': z.detach(), 'label':y_idx})
        # if init:
        #     k_means = KMeans(n_clusters=args.K).fit(z.detach(), y_idx)
        #     init = False
        # else:
        #     k_means = KMeans(n_clusters=args.K, init=k_means.cluster_centers_).fit(z.detach(), y_idx)
        # K_cluster_centrs.append(k_means.cluster_centers_)
init = True
list_of_embs = []
for emb in list_of_embs_labels:
    embs = emb['emb']
    list_of_embs.append(embs.numpy())
concatted_embs = np.empty((0, 32))
for emb in list_of_embs:
    concatted_embs = np.vstack((concatted_embs, emb))
k_means = KMeans(n_clusters=args.K).fit(concatted_embs).cluster_centers_
D_all = torch.pdist(torch.from_numpy(k_means))
# for idx in range(len(k_means)):
#     D_solo = torch.from_numpy(k_means[idx])
#     if idx == len(k_means):
#         # if last cluster
#         D_others = k_means[:idx]
#     else:
#         # get all clusters except D_solo
#         D_others = k_means[:idx] + k_means[idx+1:]
#     for D_oth in D_others:
#         dist = torch.cdist(D_solo, torch.from_numpy(D_oth))
#         Dist_all.append(dist)


# prepare dataset
used_samples = []
unused_samples = []
with torch.no_grad():
    dists = []
    for x, y_idx in data_loader_train:
        rand_number = random.randrange(0, len(D_all))
        initial_cluster = D_all[rand_number]
        z = model.forward(x)
        k_means = KMeans(n_clusters=args.K).fit_predict(z)
        dist = torch.pairwise_distance(k_means, z)
        for cluster in D_all:
            dist = cdist(cluster, k_random)
            np.fill_diagonal(dist, np.inf)
            dists.append(dist)
#    How to find the closest clusters with regard to k_random ?
# Training epoch

for epoch in range(1, 10):

    metrics_epoch = {key: [] for key in metrics.keys()}

    np_z_embs = []
    np_y_embs = []
    idx = 0
    for data_loader in [data_loader_train, data_loader_test]:  # just for classification example
        # stage = 'train'

        if data_loader == data_loader_test:
            continue
            stage = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            model = model.train()
            torch.set_grad_enabled(True)

        for x, y_idx in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_idx = y_idx.squeeze().to(DEVICE)
            z = model.forward(x)
            if data_loader == data_loader_train:
                loss, all_losses = loss_fn.forward(z=z,y_idx=y_idx)
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        print(np.mean(metrics_epoch['train_loss']))
