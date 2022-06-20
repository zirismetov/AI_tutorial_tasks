import argparse # pip3 install argparse
import hashlib
import os
import pickle
import time
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from scipy.spatial.distance import cdist
from tqdm import tqdm # pip install tqdm

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import sklearn.decomposition
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from IPython import embed

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='', type=str)

parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=66, type=int)
parser.add_argument('-classes_count', default=20, type=int)
parser.add_argument('-samples_per_class', default=600, type=int)

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

CLASS_RATIO = 1.5

DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
MAX_CLASSES = args.classes_count # 0 = include all
IS_DEBUG = args.is_debug

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

if not torch.cuda.is_available() or IS_DEBUG:
    MAX_LEN = 300 # per class for debugging
    MAX_CLASSES = 6 # reduce number of classes for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 66


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
        np_x = np.transpose(np.array(pil_x)).astype(np.float32)
        np_x = np.expand_dims(np_x, axis=0) # (1, W, H) => (1, 28, 28)
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
        idxes_free = np.random.permutation(len(self.dataset)).tolist()
        labels_free = list(self.labels_y)
        idxes_used = []
        idx_label = 0
        while len(labels_free):
            if len(labels_free) <= idx_label:
                idx_label = 0
            y_cur = labels_free[idx_label]
            idx_label += 1

            choices = []
            for idx in idxes_free:
                _, y = self.dataset[idx]
                if y_cur == y:
                    choices.append(idx)
                    if len(choices) == 3: # it could be more, batch_size should divide by this number
                        for choice_idx in choices:
                            idxes_used.append(choice_idx)
                            idxes_free.remove(choice_idx)
                        break

            if len(choices) < 3:
                labels_free.remove(y_cur)
        self.idxes_used = idxes_used

    def __len__(self):
        return (len(self.idxes_used) // self.batch_size) - 1

    def __iter__(self): # every enter => for each in DataLoaderTriplet(...)
        self.idx_batch = 0
        return self

    def __next__(self): # every iteration => for each in DataLoaderTriplet(...)
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
        z_raw = self.encoder.forward(x.expand(x.size(0), 3, 28, 28))
        z = F.normalize(z_raw, p=2, dim=-1)
        return z


class StaticProxyNCA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proxies = torch.nn.Parameter(
            torch.FloatTensor(len(labels_train), Z_SIZE)
        )
        torch.nn.init.uniform_(self.proxies)

    def forward(self, z, y_idx):
        prxies = F.normalize(self.proxies, p=2, dim=-1)
        y_rel = []
        for y_j in y_idx:
            y_j_rel = (y_to_relative_y_train == y_j).nonzero().item()
            y_rel.append(y_j_rel)
        y_rel = torch.LongTensor(y_rel)
        losses = []
        for i in range (0, len(x) -3 , 3):
            y_a = y_rel[i]
            Y_all = y_rel[i:]
            Z_all = z[i:]
            Z_pos = Z_all[Y_all == y_a]
            prox_p = prxies[y_a]
            D_p_all = F.pairwise_distance(
                prox_p.expand(  Z_pos.size()),
                Z_pos
            )
            j_p = torch.argmax(D_p_all)
            D_p = D_p_all[j_p]
            z_a = Z_pos[j_p]
            D_n_all = F.pairwise_distance(
                z_a.expand(prxies.size()),
                prxies
            )
            loss = torch.mean(-torch.log(torch.exp(-D_p) / torch.sum(torch.exp(-D_n_all)) ))
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        return loss
# Y_not_a = y_rel[i + 1:]
# Z_not_a = z[i + 1:]
# Z_pos = Z_not_a[Y_not_a==y_a]
# Z_neg = Z_not_a[Y_not_a!=y_a]
# def expand_dims(var, dim=0):
#     """ Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).
#         var = torch.range(0, 9).view(-1, 2)
#         torch.expand_dims(var, 0).size()
#         # (1, 5, 2)
#     """
#     sizes = list(var.size())
#     sizes.insert(dim, 1)
#     return var.view(*sizes)
# def comparison_mask(a_labels, b_labels):
#     """Computes boolean mask for distance comparisons"""
#     return torch.eq(expand_dims(a_labels, 1),
#                     expand_dims(b_labels, 0))
# class MagnetLoss(torch.nn.Module):
#     def __init__(self, D=12, M=4, alpha=7.18):
#         super().__init__()
#
#         self.D = D
#         self.M = M
#         self.alpha = alpha
#
#     def forward(self, logits, classes, m, d, alpha=1.0):
#         GPU_INT_DTYPE = torch.cuda.IntTensor
#         GPU_LONG_DTYPE = torch.cuda.LongTensor
#         GPU_FLOAT_DTYPE = torch.cuda.FloatTensor
#         """
#         :param  indices     The index of each embedding
#         :param  outputs     The set of embeddings
#         :param  clusters    Cluster assignments for each index
#         :return Loss        Magnet loss calculated for current batch
#         """
#         self.logits = logits
#         self.classes = torch.from_numpy(classes).type(GPU_LONG_DTYPE)
#         self.clusters, _ = torch.sort(torch.arange(0, float(m)).repeat(d))
#         self.clusters = self.clusters.type(GPU_INT_DTYPE)
#         self.cluster_classes = self.classes[0:m * d:d]
#         self.n_clusters = m
#         self.alpha = alpha
#
#         # Take cluster means within the batch
#         cluster_examples = torch.chunk(self.logits, self.n_clusters)
#
#         cluster_means = torch.stack([torch.mean(x, dim=0) for x in cluster_examples])
#
#         sample_costs = F.pairwise_distance(cluster_means, expand_dims(logits, 1))
#
#         clusters_tensor = self.clusters.type(GPU_FLOAT_DTYPE)
#         n_clusters_tensor = torch.arange(0, self.n_clusters).type(GPU_FLOAT_DTYPE)
#
#         intra_cluster_mask = torch.autograd.Variable(comparison_mask(clusters_tensor, n_clusters_tensor).type(GPU_FLOAT_DTYPE))
#
#         intra_cluster_costs = torch.sum(intra_cluster_mask * sample_costs, dim=1)
#
#         N = logits.size()[0]
#
#         variance = torch.sum(intra_cluster_costs) / float(N - 1)
#
#         var_normalizer = -1 / (2 * variance ** 2)
#
#         # Compute numerator
#         numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)
#
#         classes_tensor = self.classes.type(GPU_FLOAT_DTYPE)
#         cluster_classes_tensor = self.cluster_classes.type(GPU_FLOAT_DTYPE)
#
#         # Compute denominator
#         diff_class_mask = torch.autograd.Variable(comparison_mask(classes_tensor, cluster_classes_tensor).type(GPU_FLOAT_DTYPE))
#
#         diff_class_mask = 1 - diff_class_mask  # Logical not on ByteTensor
#
#         denom_sample_costs = torch.exp(var_normalizer * sample_costs)
#
#         denominator = torch.sum(diff_class_mask * denom_sample_costs, dim=1)
#
#         epsilon = 1e-8
#
#         losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))
#
#         total_loss = torch.mean(losses)
#
#         return total_loss, losses

class DynamicProxyNCA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proxies = torch.nn.Parameter(
            torch.FloatTensor(int(len(labels_train) * CLASS_RATIO), Z_SIZE)
        )
        torch.nn.init.uniform_(self.proxies)

    def forward(self, z, y_idx):
        prxies = F.normalize(self.proxies, p=2, dim=-1)
        y_rel = []
        for y_j in y_idx:
            y_j_rel = (y_to_relative_y_train == y_j).nonzero().item()
            y_rel.append(y_j_rel)
        y_rel = torch.LongTensor(y_rel)
        losses = []
        for i in range(0, len(x) - 3, 3):
            y_a = y_rel[i]
            Y_all = y_rel[i:]
            Z_all = z[i:]
            Z_p = Z_all[Y_all == y_a]
            z_a = z[i]
            D_prox = F.pairwise_distance(
                z_a.expand(prxies.size()),
                prxies
            )
            y_prox = torch.argmin(D_prox)

            prox_p = prxies[y_prox]
            D_p_all = F.pairwise_distance(
                prox_p.expand(Z_p.size()),
                Z_p
            )
            j_p = torch.argmax(D_p_all)
            D_p = D_p_all[j_p]
            z_a = Z_p[j_p]
            D_n_all = F.pairwise_distance(
                z_a.expand(prxies.size()),
                prxies
            )
            loss = torch.mean(-torch.log(torch.exp(-D_p) / torch.sum(torch.exp(-D_n_all))))
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        return loss


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = StaticProxyNCA().to(DEVICE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'x', 'z', 'y', 'acc']:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

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

            if data_loader == data_loader_train:
                loss = loss_fn.forward(z, y_idx, m=12, d=4)
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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
        if key not in ['train_z', 'train_y', 'train_x', 'test_z', 'test_y', 'test_x', 'test_loss']:
            value = 0
            if len(metrics_epoch[key]):
                value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')


    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf() # BUGFIX!! scatter is remembered even when new subplot is made

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        if key in ['train_z', 'train_y', 'train_x', 'test_z', 'test_y', 'test_x', 'test_loss']:
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
    scatter = plt.scatter(np_z[:MAX_POINTS, 0], np_z[:MAX_POINTS, 1], c=c, cmap=plt.get_cmap('tab20'))
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
    scatter = plt.scatter(np_z[:MAX_POINTS, 0], np_z[:MAX_POINTS, 1], c=c, cmap=plt.get_cmap('tab20'))
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
        if np.isnan(metrics[f'train_loss'][-1]):
            exit()
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')

