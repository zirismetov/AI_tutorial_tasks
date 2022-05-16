# %%writefile hard_mining.py
import argparse
import hashlib
import os
import pickle
import time
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from scipy.spatial.distance import cdist
from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import random
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')
from copy import copy

import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)
parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-classes_count', default=20, type=int)
parser.add_argument('-samples_per_class', default=10000, type=int)
parser.add_argument('-run_name', default=f'run', type=str)

parser.add_argument('-learning_rate', default=1e-4, type=float)

parser.add_argument('-z_size', default=32, type=int)
parser.add_argument('-margin', default=0.8, type=float)

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
NOW_TIME = datetime.strftime(datetime.today(), "%Y-%m-%d_%H-%M")

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
else:
    RUN_PATH = f'/content/drive/MyDrive/zafar_iris_folder/hard_mining/{NOW_TIME}'
os.makedirs(RUN_PATH)

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 100 # per class for debugging
    MAX_CLASSES = 10 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 12

class TensorBoardSummaryWriter(SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=10, filename_suffix='',
                 write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue,
                         flush_secs, filename_suffix, write_to_disk,
                         log_dir, **kwargs)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        hparam_dict = copy(hparam_dict)
        for key in list(hparam_dict.keys()):
            if type(hparam_dict[key]) not in [float, int, str]:
                del hparam_dict[key]
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)


summary_writer = TensorBoardSummaryWriter(
    logdir=RUN_PATH + '/tenorboard'
)
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

print(
    f'labels_train: {labels_train} '
    f'labels_test: {labels_test} '
)
idx_train = []
idx_test = []
str_args = [str(it) for it in [MAX_LEN, MAX_CLASSES]]
hash_args = hashlib.md5((''.join(str_args)).encode()).hexdigest()
path_cache = f'{RUN_PATH}/cache_pkl/{hash_args}.pkl'
if os.path.exists(path_cache):
    print('loading from cache')
    with open(path_cache, 'rb') as fp:
        idx_train, idx_test = pickle.load(fp)

else:
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

    # with open(path_cache, 'wb') as fp:
    #     pickle.dump((idx_train, idx_test), fp)

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
        self.encoder = torchvision.models.resnet18( # 3, W, H
            pretrained=True
        )
        self.encoder.fc = torch.nn.Linear(
            in_features=self.encoder.fc.in_features,
            out_features=Z_SIZE
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(Z_SIZE, len(labels_train)),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        z = self.encoder.forward(x.expand(x.size(0), 3, 28, 28))
        z = z.view(-1, Z_SIZE)
        y_prim = self.classifier.forward(z)
        return y_prim, z

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'z', 'y', 'acc', 'dist_pos', 'dist_neg']:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(EPOCHS):
    metrics_epoch = {key: [] for key in metrics.keys()}
    for data_loader in [data_loader_train, data_loader_test]: # just for classification example
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
        logits_with_labels_and_images = []
        for x, y_idx in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_idx = y_idx.squeeze().to(DEVICE)
            y_idx_copy = torch.clone(y_idx).to(DEVICE)
            y_prim, z = model.forward(x)
            for idx in range(len(y_idx_copy)):
                y_label = dataset_full.labels[int(y_idx_copy[idx].item())]
                if stage == 'train':
                    y_idx_copy[idx] = labels_train.index(y_label)
                else:
                    y_idx_copy[idx] = labels_test.index(y_label)
            CCE_loss = torch.mean(torch.log(y_prim[:, y_idx_copy[range(len(y_idx_copy))]] + 1e-8))
            dist_pos = []
            dist_neg = []
            losses = []
            for i, y_idx_i in enumerate(y_idx):
                D_all = 1.0 - F.cosine_similarity(
                    z[i].expand(z.size()),
                    z
                )
                D_neg_all = D_all[y_idx != y_idx_i]
                j = torch.argmin(D_neg_all)
                D_n = D_neg_all[j]
                dist_neg += D_neg_all.cpu().data.numpy().tolist()
                D_pos_all = D_all[y_idx == y_idx_i]

                if len(D_pos_all) > 1 :
                    j = torch.argmax(D_pos_all)
                    D_p = D_pos_all[j]
                    c_losses = D_p + torch.maximum(MARGIN - D_n, torch.zeros_like(D_n))
                    losses.append(c_losses)
                    D_pos_all = D_pos_all[D_pos_all != 0]
                    dist_pos += D_pos_all.cpu().data.numpy().tolist()

            loss = torch.mean(torch.stack(losses))
            metrics_epoch[f'{stage}_dist_pos'].append(np.mean(dist_pos))
            metrics_epoch[f'{stage}_dist_neg'].append(np.mean(dist_neg))
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_z = z.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y_idx = y_idx.cpu().data.numpy()
            np_y_prim = y_prim.cpu().data.numpy()

            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_y'] += np_y_idx.tolist()
            for idx in range(len(np_y_idx)):
                if len(logits_with_labels_and_images) < 2500:
                    logits_with_labels_and_images.append((
                        np_y_prim[idx],
                        dataset_full.labels[np_y_idx[idx]],
                        x.data.cpu().numpy()[idx]
                    ))

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
                metric='cosine'
            ).squeeze()
            idx_closest = np.argmin(dists_to_centers)
            y_prim_idx_each = list(centers_by_classes.keys())[idx_closest]
            y_prim_idx.append(y_prim_idx_each)

        np_y_prim_idx = np.array(y_prim_idx)
        acc = np.mean((np_y_prim_idx == np_ys) * 1.0)
        metrics_epoch[f'{stage}_acc'] = [acc]
        embs, labels, imgs = list(zip(*logits_with_labels_and_images))
        summary_writer.add_embedding(
            mat=np.array(embs),
            metadata=list(labels),
            label_img=np.array(imgs),
            tag= f'{stage}_logits',
            global_step=epoch
        )

    metrics_strs = []
    metrics_mean = {}
    for key in metrics_epoch.keys():
        if '_z' not in key and '_y' not in key:
            value = 0
            if len(metrics_epoch[key]):
                value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_mean[key] = value
            summary_writer.add_scalar( #ADD_SCALAR is for plots
                scalar_value=value,
                tag = key,
                global_step=epoch
            )
            metrics_strs.append(f'{key}: {round(value, 2)}')

    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict=metrics_mean,
        name=args.run_name,
        global_step=epoch
    )
    summary_writer.flush()
    if os.path.exists(f'{RUN_PATH}/logs_{NOW_TIME}.txt'):
        with open(f'{RUN_PATH}/logs_{NOW_TIME}.txt', 'a') as f:
            f.write(f'epoch: {epoch} {" ".join(metrics_strs)} \n')
    else:
        with open(f'{RUN_PATH}/logs_{NOW_TIME}.txt', 'w+') as f:
            f.write(f'epoch: {epoch} {" ".join(metrics_strs)} \n')
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        if '_z' in key or '_y' in key:
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

    pca = sklearn.decomposition.KernelPCA(n_components=2)

    MAX_POINTS = 1000
    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    np_y_idx = np.array(metrics_epoch[f'train_y'])
    labels = [dataset_full.labels[idx] for idx in np_y_idx[:MAX_POINTS]]
    set_labels = list(set(labels))
    c = [labels.index(it) for it in labels]
    scatter = plt.scatter(np_z[:MAX_POINTS, -1], np_z[:MAX_POINTS, -2], c=c, cmap=plt.get_cmap('prism'))
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc='lower left',
        scatterpoints=1,
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
    c = [labels.index(it) for it in labels]
    scatter = plt.scatter(np_z[:MAX_POINTS, -1], np_z[:MAX_POINTS, -2], c=c, cmap=plt.get_cmap('prism'))
    plt.legend(
        handles=scatter.legend_elements()[0],
        loc='lower left',
        scatterpoints=1,
        fontsize=8,
        ncol=5,
        labels=set_labels
    )

    plt.tight_layout(pad=0.5)


    if not IS_DEBUG:
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')
        torch.save(model.state_dict(), f'{RUN_PATH}/model-{epoch}.pt')
        plt.show()
    else:
        exit()
# else:
#     # if np.isnan(metrics[f'train_loss'][-1]) or np.isinf(metrics[f'test_loss'][-1]):
#     #     exit()

#     # save model weights
#     plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')

