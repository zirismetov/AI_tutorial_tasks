import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import random

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import scipy.misc
import scipy.ndimage
from scipy.spatial.distance import  cdist
import sklearn.decomposition
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
TRAIN_TEST_SPLIT = 0.8
DEVICE = 'cuda'
MAX_LEN = 0
MAX_CLASSES = 0
IS_DEBUG = True

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 100 # per class for debugging
    MAX_CLASSES = 10 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 12


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() # 62 classes
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
        np_x = np.array(pil_x)
        np_x = np.expand_dims(np_x, axis=0) # (1, W, H)

        x = torch.FloatTensor(np_x)
        y = torch.LongTensor([y_idx])
        return x, y


dataset_full = DatasetEMNIST()

labels_train, labels_test = train_test_split(
    dataset_full.labels[-MAX_CLASSES:],
    train_size=TRAIN_TEST_SPLIT,
    shuffle=True,
    random_state=0
)
labels_count = dict((key, 0) for key in dataset_full.labels)

print(f'labels_test: {labels_test} labels_train: {labels_train}')

idx_train = []
idx_test = []
for idx, (x, y_idx) in tqdm(enumerate(dataset_full), 'splitting dataset', total=len(dataset_full)):
    y_idx = y_idx.item()
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

dataset_train = torch.utils.data.Subset(dataset_full, idx_train)
dataset_test = torch.utils.data.Subset(dataset_full, idx_test)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE < 12)
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_test) % BATCH_SIZE < 12)
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True)

        self.encoder.fc = torch.nn.Linear(in_features=self.encoder.fc.in_features,
                                          out_features=32)

    def forward(self, x):
        z = self.encoder.forward(x.expand(BATCH_SIZE, 3, 28, 28))
        z = z.view(-1, 32)
        y_prim = torch.zeros_like(z)
        return y_prim, z


model = Model()

# TODO use dummy code to check model
# dummy = torch.randn((BATCH_SIZE, 1, 28, 28))
# y_target = model.forward(dummy)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'z', 'y', 'acc']:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]:
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y_idx in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_idx = y_idx.squeeze().to(DEVICE)
            y_prim, z = model.forward(x)
            if data_loader == data_loader_train:
                D_n = []
                D_p = []
                for i in range(len(y_idx)):
                    for j in range(i+1, len(y_idx)):
                        z_a = z[i]
                        y_a = y_idx[i]
                        if y_a == y_idx[j]:
                            z_p = z[j]
                            D_p.append((z_a-z_p)**2)
                        else:
                            z_n = z[j]
                            D_n.append((z_a-z_n)**2)
                D_n = torch.stack(D_n)
                D_p = torch.stack(D_p)
                loss = torch.mean(D_p) + torch.mean(torch.relu(0.2 - D_n))
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            np_y_prim = y_prim.cpu().data.numpy()
            np_z = z.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y_idx = y_idx.cpu().data.numpy()
            np_y_prim_idx = np.argmax(np_y_prim, axis=-1)


            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_y'] += np_y_idx.tolist()
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
                metric= 'euclidean'
            ).squeeze()
            idx_closest = np.argmin(dists_to_centers)
            y_prim_idx_each = list(centers_by_classes.keys())[idx_closest]
            y_prim_idx.append(y_prim_idx_each)
        y_prim_idx = np.array(y_prim_idx)
        acc = np.mean((y_prim_idx == np_ys) * 1.0)
        metrics_epoch[f'{stage}_acc'].append(acc)
    metrics_strs = []
    for key in metrics_epoch.keys():
        if '_z' not in key and '_y' not in key:
            value = 0
            if len(metrics_epoch[key]):
                value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        if '_z' in key or '_y' in key or 'test_' in key:
            continue
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18, 10, 11, 12, 22, 23, 24]):
        plt.subplot(8, 6, j) # row col idx
        color = 'green' if np_y_idx[i] == np_y_prim_idx[i] else 'red'
        plt.title(f"y: {dataset_full.labels[np_y_idx[i]]} y_prim: {dataset_full.labels[np_y_prim_idx[i]]}", color=color)
        plt.imshow(np.transpose(np_x[i][0]), cmap='Greys')

    plt.subplot(223) # row col idx

    pca = sklearn.decomposition.PCA(n_components=2, whiten=True)

    MAX_POINTS = 1000
    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    np_y_idx = np.array(metrics_epoch[f'train_y'])
    labels = [dataset_full.labels[idx] for idx in np_y_idx[:MAX_POINTS]]
    set_labels = list(set(labels))
    c = np.array([set_labels.index(it) for it in labels])
    scatter = plt.scatter(np_z[:MAX_POINTS, -1], np_z[:MAX_POINTS, -2], c=c, cmap=plt.get_cmap('prism'))
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc='lower left',
        fontsize=8,
        ncol=5,
        labels=set_labels
    )


    plt.subplot(224) # row col idx

    plt.title('test_z')
    np_z = np.array(metrics_epoch[f'test_z'])
    np_z = pca.fit_transform(np_z)
    np_y_idx = np.array(metrics_epoch[f'test_y'])
    labels = [dataset_full.labels[idx] for idx in np_y_idx[:MAX_POINTS]]
    set_labels = list(set(labels))
    c = np.array([set_labels.index(it) for it in labels])
    scatter = plt.scatter(np_z[:MAX_POINTS, -1], np_z[:MAX_POINTS, -2], c=c, cmap=plt.get_cmap('prism'))
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc='lower left',
        fontsize=8,
        ncol=5,
        labels=set_labels
    )

    plt.tight_layout(pad=0.5)
    plt.show()
