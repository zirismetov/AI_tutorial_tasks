import functools
import random
from functools import reduce

import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

MAX_LEN = 200 # For debugging, reduce number of samples
BATCH_SIZE = 16
n_classes = 0
n_names = np.array([])

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        global n_classes
        global n_names

        super().__init__()
        self.data = fetch_lfw_people(min_faces_per_person=20, color=True)
        n_classes = self.data.target_names.size
        n_names = self.data.target_names


    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        np_x = self.data.images[idx]
        # np_x = np.expand_dims(np_x, axis=0)
        x = torch.FloatTensor(np_x)
        x = torch.movedim(x, 2, 0)

        np_y = np.zeros((n_classes,))
        np_y[self.data.target[idx]] = 1
        y = torch.FloatTensor(np_y)

        return x, y


dataset = LoadDataset()
devide_by_idx = np.arange(len(dataset))
subset_train_data, subset_test_data = train_test_split(
    devide_by_idx,
    test_size=0.2,
    random_state=8)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)



class Shuffle(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups,
            channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class ShuffleNetBlock(torch.nn.Module):
    def __init__(self, in_features, num_groups):
        super().__init__()


        # Shuffle is given above
        # DW convolution is Conv2D where in_features == groups
        # GConvolution id Conv2D with groups parameter

        self.chain = torch.nn.Sequential(

            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=num_groups
                ),

            torch.nn.BatchNorm2d(in_features),
            torch.nn.ReLU(),

            Shuffle(num_groups),

            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=num_groups,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=num_groups
            ),
            torch.nn.BatchNorm2d(num_groups),

            torch.nn.Conv2d(
                in_channels=num_groups,
                out_channels=in_features,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=num_groups
            ),
            torch.nn.BatchNorm2d(in_features),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.chain.forward(x)
        y_prim = out + x
        return y_prim


class Reshape(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    def forward(self, x):
        return x.view(self.target_shape)

class ShuffleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 32
        num_groups = 4
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=num_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            torch.nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            ShuffleNetBlock(in_features=num_channels, num_groups=num_groups),
            torch.nn.AdaptiveAvgPool2d(output_size=1),  #(B, num_channels, 1, 1)
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=n_classes),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layers.forward(x)

model = ShuffleNet()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in tqdm.tqdm(range(1, 500)):
    plt.clf()

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = model.forward(x)

            loss = torch.sum(-y*torch.log(y_prim + 1e-8))
            # Sum dependant on batch size => larger LR
            # Mean independant of batch size => smaller LR

            # y.to('cuda')
            # y.cuda()

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.mean((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 3)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        value = gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig('plot_shuffle_lfw')
plt.savefig('plot_shuffle_lfw')


