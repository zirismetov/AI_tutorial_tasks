import sklearn.metrics
import torchvision.models
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional
import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.datasets import fetch_lfw_people, make_blobs
from sklearn.model_selection import train_test_split
import time


MAX_LEN = 200
INPUT_SIZE_HEIGHT = 62
INPUT_SIZE_WIDTH = 47
DEVICE = 'cpu'
n_classes = 0
n_names = np.array([])
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

LEARNING_RATE = 1e-4
BATCH_SIZE = 16

# summary_writer = tensorboard_utils.CustomSummaryWriter(
#     logdir=f'{args.seq_name}/{args.run_name}'
# )

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        global n_classes
        global n_names

        super().__init__()
        self.data = fetch_lfw_people(color=True)
        n_classes = self.data.target_names.size
        n_names = self.data.target_names
        pass


    def __len__(self):
        if DEVICE == 'cpu':
            return MAX_LEN
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
    random_state=22)

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

# tensorboard --logdir=./seq_name

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=512,
            out_features=n_classes
        )

    def forward(self, x):
        x_fun = self.encoder.forward(x)
        x_fun = x_fun.view(x_fun.size(0), -1)
        logits = self.fc.forward(x_fun)
        y_prim = torch.softmax(logits, dim=1)

        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc',
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in tqdm.tqdm(range(1, 200)):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)
            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

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
        plt.savefig('plot_vgg_lfw')
