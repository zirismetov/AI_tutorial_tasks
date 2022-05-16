# %%writefile c_loss_tensorboard.py
import datetime
import os
import pickle
import time

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import random
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
import warnings
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import torchvision
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
import argparse
from datetime import datetime
from copy import copy
from torchvision.datasets import EMNIST


parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-run_name', default=f'run', type=str)
parser.add_argument('-sequence_name', default=f'seq_{datetime.strftime(datetime.now(), "%m-%d_%H-%M-%S")}', type=str)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=50, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('-emb_size', default=8, type=int)
parser.add_argument('-max_len', default=300, type=int)
parser.add_argument('-max_classes', default=20, type=int)
parser.add_argument('-is_debug', default=True, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()

TRAIN_TEST_SPLIT = 0.8
MARGIN = 0.2
COEF_CCE = 0.5

if not torch.cuda.is_available() or args.is_debug:
    print('Using debug version without GPU')
    args.max_len = 100 # per class for debugging
    args.max_classes = 10 # reduce number of classes for debugging
    args.device = 'cpu'
    args.batch_size = 12


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() # 62 classes
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            split='byclass',
            train=(args.max_len == 0),
            download=True,
            transform=torchvision.transforms.Resize(56)
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
    dataset_full.labels[-args.max_classes:],
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
for idx, (x, y_idx) in tqdm(enumerate(dataset_full), 'splitting dataset', total=len(dataset_full)):
    y_idx = y_idx.item()
    label = dataset_full.labels[y_idx]
    if args.max_len > 0: # for debugging
        if labels_count[label] >= args.max_len:
            if all(it >= args.max_len for it in labels_count.values()):
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
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=(len(dataset_train) % args.batch_size < 12)
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=(len(dataset_test) % args.batch_size < 12)
)

def convert_y_to_train_y(y):
    y_relative = torch.zeros_like(y)
    for i, y_i in enumerate(y):
        y_i_int = int(y_i.item())
        y_i_label = dataset_full.labels[y_i_int]
        y_relative[i] = labels_train.index(y_i_label)
    return y_relative


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.densenet121(pretrained=True)
        self.encoder.classifier = torch.nn.Linear(in_features=self.encoder.classifier.in_features,
                                                  out_features=args.emb_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size, len(labels_train)),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        z = self.encoder.forward(x.expand(-1, 3, 56, 56))
        z = z.view(-1, args.emb_size)
        y_prim = self.classifier.forward(z)
        return y_prim, z


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
model = model.to(args.device)
RUN_PATH = f'/content/drive/MyDrive/zafar_iris_folder/c_loss_tensorboard/{args.sequence_name}/{args.run_name}_{int(time.time())}'
if not os.path.exists(RUN_PATH):
    os.makedirs(RUN_PATH)
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
metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'z', 'y', 'acc']:
        metrics[f'{stage}_{metric}'] = []
contrastiveLoss_fn = losses.ContrastiveLoss(neg_margin=MARGIN, distance=CosineSimilarity())
with warnings.catch_warnings():
    warnings.simplefilter("error", category=RuntimeWarning)
for epoch in tqdm(range(args.epochs)):
    metrics_epoch = {key: [] for key in metrics.keys()}
    for data_loader in [data_loader_train, data_loader_test]: # just for classification example
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
        logits_with_labels_and_images = []
        for x, y_idx in data_loader:
            x = x.to(args.device)
            y_idx = y_idx.squeeze().to(args.device)
            y_prim, z = model.forward(x)

            if data_loader == data_loader_train:
                contrastiveLoss = contrastiveLoss_fn(z, y_idx)
                for idx in range(len(y_idx)):
                    y_label = dataset_full.labels[int(y_idx[idx].item())]
                    y_idx[idx] = labels_train.index(y_label)
                CCE_loss = torch.mean(torch.log(y_prim[:, y_idx[range(len(y_idx))]] + 1e-8))
                loss = contrastiveLoss - COEF_CCE*CCE_loss
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
            for idx in range(len(np_y_idx)):
                # if len(logits_with_labels_and_images) < 1000:
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

        y_prim_idx = np.array(y_prim_idx)
        acc = np.mean((y_prim_idx == np_ys) * 1.0)
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
            metrics_mean[key] = value
            summary_writer.add_scalar( #ADD_SCALAR is for plots
                scalar_value=value,
                tag = key,
                global_step=epoch
            )
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')
    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict=metrics_mean,
        name=args.run_name,
        global_step=epoch
    )
    summary_writer.flush()
    if epoch % 10 == 0 or epoch == (args.epochs-1):
        plt.clf()

        plt.subplot(221) # row col idx
        plts = []
        c = 0
        for key, value in metrics.items():
            if '_z' in key or '_y' in key or key=='test_loss':
                continue
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        np_y_prim_idx = y_prim_idx[-12:]
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
        c = [labels.index(it) for it in labels]
        scatter = plt.scatter(np_z[:MAX_POINTS, -1], np_z[:MAX_POINTS, -2], c=c, cmap=plt.get_cmap('tab20c'))
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
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')
        plt.show()


