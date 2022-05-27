import argparse # pip3 install argparse
import hashlib
import os
import pickle
import time
import shutil
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from scipy.spatial.distance import cdist
from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import random
import sklearn.manifold

from tensorboardX import SummaryWriter


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='exp', type=str)

parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-classes_count', default=20, type=int)
parser.add_argument('-samples_per_class', default=1000, type=int)

parser.add_argument('-learning_rate', default=1e-4, type=float)

parser.add_argument('-z_size', default=32, type=int)
parser.add_argument('-margin', default=0.2, type=float)

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

DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count # 0 = include all
IS_DEBUG = args.is_debug

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

    tensorboard_writer = SummaryWriter(f'{RUN_PATH}/tb_{RUN_PATH}')

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 300 # per class for debugging
    MAX_CLASSES = 6 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 66

assert BATCH_SIZE % 4 == 0

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='byclass',
            train=(MAX_LEN == 1000),
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
        np_x = np.transpose(np.array(pil_x)).astype(np.float16)
        np_x = np.expand_dims(np_x, axis=0) # (1, W, H)
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
        if MAX_LEN > 0: # for debugging
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

class DataLoaderTriplet():
    def __init__(
            self,
            dataset,
            batch_size,
            labels_y
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels_y = labels_y
        self.idxes_used = []
        self.reshuffle()

    def reshuffle(self):
        idx_free = np.random.permutation(len(self.dataset)).tolist()
        label_free =  list(self.labels_y)
        idx_used = []
        idx_label = 0

        while len(label_free):
            if len(label_free) <= idx_label:
                idx_label = 0
            y_cur = label_free[idx_label]
            idx_label += 1
            choices = []
            for idx in idx_free:
                _, y = self.dataset[idx]
                if y_cur == y:
                    choices.append(idx)
                    if len(choices) == 4:
                        for choice_idx in choices:
                            idx_used.append(choice_idx)
                            idx_free.remove(choice_idx)
                        break
            if len(choices)<4:
                label_free.remove(y_cur)
        self.idxes_used = idx_used



    def __len__(self):
        return (len(self.idxes_used) // self.batch_size) - 1

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch > len(self):
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size
        idx_end = idx_start + self.batch_size
        x_list = []
        y_list = []
        for idx in range(idx_start, idx_end):
            idx_mapped = self.idxes_used[idx]
            x, y = self.dataset[idx_mapped]
            x_list.append(torch.FloatTensor(x))
            y_list.append(torch.LongTensor([y]))
        x = torch.stack(x_list)
        y = torch.stack(y_list).squeeze()
        self.idx_batch += 1
        return x, y


data_loader_train = DataLoaderTriplet(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    labels_y=labels_y_train
)

data_loader_test = DataLoaderTriplet(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    labels_y=labels_y_test
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet18( # 3, W, H
            pretrained=False
        )
        self.encoder.fc = torch.nn.Linear(
            in_features=self.encoder.fc.in_features,
            out_features=Z_SIZE
        )
    def forward(self, x):
        z = self.encoder.forward(x.expand(x.size(0), 3, 28, 28))
        norm = torch.norm(z.detach(), p=2, dim=-1, keepdim=True)
        z = z / norm
        return z

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'x', 'z', 'y', 'acc', 'dist_pos', 'dist_neg']:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 15):

    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]: # just for classification example
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            if epoch > 1:
                data_loader.reshuffle()
            model = model.train()
            torch.set_grad_enabled(True)

        for x, y_idx in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_idx = y_idx.squeeze().to(DEVICE)
            z = model.forward(x)

            dist_pos = []
            dist_neg = []
            losses = []

            for i in range(0, len(y_idx)-4, 4):
                y_idx_i = y_idx[i]
                z_not_used = z[i+1:]
                y_not_used = y_idx[i+1:]

                D_all = F.pairwise_distance(
                    z[i].expand(z_not_used.size()),
                    z_not_used,
                    p=2
                )
                D_neg_all = D_all[y_not_used != y_idx_i]
                j = torch.argmin(D_neg_all)
                D_n = D_neg_all[j]

                D_pos_all = D_all[y_not_used == y_idx_i]
                j = torch.argmax(D_pos_all)
                D_p = D_pos_all[j]
                if D_p<D_n:
                    dist_neg += D_neg_all.cpu().data.numpy().tolist()
                    dist_pos += D_pos_all.cpu().data.numpy().tolist()
                    losses.append(torch.relu(D_p-D_n + MARGIN))
            if losses:
                loss = torch.mean(torch.stack(losses))
                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                metrics_epoch[f'{stage}_dist_pos'].append(np.mean(dist_pos))
                metrics_epoch[f'{stage}_dist_neg'].append(np.mean(dist_neg))
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        torch.set_grad_enabled(False)
        model = model.eval()
        for x, y_idx in tqdm(data_loader, desc=f'acc_{stage}'):
            x = x.to(DEVICE)
            z = model.forward(x)
            np_z = z.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y_idx = y_idx.squeeze().cpu().data.numpy()
            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_y'] += np_y_idx.tolist()
            metrics_epoch[f'{stage}_x'] += np_x.tolist()

        # calculate centers of the mass
        np_zs = np.array(metrics_epoch[f'{stage}_z'])
        np_ys = np.array(metrics_epoch[f'{stage}_y'])
        centers_by_classes = {}
        for y_idx in set(np_ys):
            matching_zs = np_zs[np_ys == y_idx]
            center = np.mean(matching_zs, axis=0)
            centers_by_classes[y_idx] = center

        y_prim_idx = []
        for z in np_zs:
            dists_to_centers = cdist(
                np.array(list(centers_by_classes.values())),
                np.expand_dims(z, axis=0),
                metric='euclidean'
            ).squeeze()
            idx_closest = np.argmin(dists_to_centers)
            y_prim_idx_each = list(centers_by_classes.keys())[idx_closest]
            y_prim_idx.append(y_prim_idx_each)

        np_y_prim_idx = np.array(y_prim_idx)
        acc = np.mean((np_y_prim_idx == np_ys) * 1.0)
        metrics_epoch[f'{stage}_acc'] = [acc]



    metrics_strs = []
    for key in metrics_epoch.keys():
        if '_z' not in key and '_y' not in key:
            value = 0
            if len(metrics_epoch[key]):
                value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')
            tensorboard_writer.add_scalar(tag=key, scalar_value=value, global_step=epoch)
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf() # BUGFIX!! scatter is remembered even when new subplot is made

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        if '_z' in key or '_y' in key or '_dist' in key or 'test_' in key:
            continue
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18, 10, 11, 12, 22, 23, 24]):
        plt.subplot(8, 6, j) # row col idx
        color = 'green' if np_y_idx[i] == np_y_prim_idx[i] else 'red'
        plt.title(f"y: {dataset_full.labels[np_y_idx[i]]} y_prim: {dataset_full.labels[np_y_prim_idx[i]]}", color=color)
        plt.imshow(np_x[i][0], cmap='Greys')

    plt.subplot(223) # row col idx


    pca = sklearn.decomposition.PCA(n_components=2)

    MAX_POINTS = 100
    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    np_y_idx = np.array(metrics_epoch[f'train_y'])
    labels = [dataset_full.labels[idx] for idx in np_y_idx[:MAX_POINTS]]
    set_labels = list(set(labels))
    c = [labels.index(it) for it in labels]
    scatter = plt.scatter(np_z[:MAX_POINTS, 0], np_z[:MAX_POINTS, 1], c=c, cmap=plt.get_cmap('rainbow'))
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc='lower left',
        scatterpoints=1,
        fontsize=8,
        ncol=5,
        labels=set_labels
    )

    pca = sklearn.decomposition.PCA(n_components=2)

    plt.subplot(224) # row col idx

    plt.title('test_z')
    np_z = np.array(metrics_epoch[f'test_z'])
    np_z = pca.fit_transform(np_z)
    np_y_idx = np.array(metrics_epoch[f'test_y'])
    labels = [dataset_full.labels[idx] for idx in np_y_idx[:MAX_POINTS]]
    set_labels = list(set(labels))
    c = [labels.index(it) for it in labels]
    scatter = plt.scatter(np_z[:MAX_POINTS, 0], np_z[:MAX_POINTS, 1], c=c, cmap=plt.get_cmap('rainbow'))
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc='lower left',
        scatterpoints=1,
        fontsize=8,
        ncol=5,
        labels=set_labels
    )

    plt.tight_layout(pad=0.5)


    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss'][-1]) or np.isinf(metrics[f'test_loss'][-1]):
            exit()
        np_z =np.array(metrics_epoch['train_z'])
        np_x =np.array(metrics_epoch['train_x'])
        np_y =np.array(metrics_epoch['train_y'])
        labels = [dataset_full.labels[it] for it in np_y]
        tensorboard_writer.add_embedding(
            np_z,
            labels,
            label_img=np_x,
            global_step=epoch,
            tag='train_embs'
        )
        tensorboard_writer.flush()

        # save model weights
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')
        #torch.save(model.state_dict(), f'{RUN_PATH}/model-{epoch}.pt')

