from sklearn.datasets import fetch_lfw_people
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional
from sklearn.model_selection import train_test_split

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
MAX_LEN = 200
INPUT_SIZE_HEIGHT = 62
INPUT_SIZE_WIDTH = 47
DEVICE = 'cpu'
n_classes = 0
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


def get_out_size(in_ch, pad, kernel, stride):
    return int((in_ch + 2 * pad - kernel) / stride + 1)


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        global n_classes
        super().__init__()
        self.data = fetch_lfw_people(min_faces_per_person=80)
        n_classes = self.data.target_names.size


    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        np_x = self.data.images[idx]
        np_x = np.expand_dims(np_x, axis=0)
        x = torch.FloatTensor(np_x)

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


class Conv2d(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = torch.nn.Parameter(
            torch.FloatTensor(kernel_size, kernel_size, in_channels, out_channels)
        )

        torch.nn.init.xavier_uniform_(self.K)

    def forward(self, x):
        batch_size = x.size(0)
        in_h_size = x.size(2)
        in_w_size = x.size(3)

        out_size_h = get_out_size(in_h_size, self.padding, self.kernel_size, self.stride)
        out_size_w = get_out_size(in_w_size, self.padding, self.kernel_size, self.stride)

        out = torch.zeros(batch_size, self.out_channels, out_size_h, out_size_w).to(DEVICE)

        x_padded = torch.zeros(batch_size, self.in_channels, in_h_size + self.padding * 2,
                               in_w_size + self.padding * 2).to(DEVICE)
        x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x

        K = self.K.view(-1, self.out_channels)

        i_out = 0
        for i in range(0, out_size_h - self.kernel_size, self.stride):
            j_out = 0
            for j in range(0, out_size_w - self.kernel_size, self.stride):
                x_part = x_padded[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                x_part = x_part.reshape(batch_size, -1)

                out_part = x_part @ K
                out[:, :, i_out, j_out] = out_part
                j_out += 1
            i_out += 1

        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            Conv2d(in_channels=1,
                   out_channels=3,
                   kernel_size=5,
                   stride=1,
                   padding=1),
            torch.nn.LeakyReLU(),
            Conv2d(in_channels=3,
                   out_channels=6,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            torch.nn.LeakyReLU(),
            Conv2d(in_channels=6,
                   out_channels=12,
                   kernel_size=3,
                   stride=1,
                   padding=1),
        )

        out_1_h = get_out_size(INPUT_SIZE_HEIGHT, kernel=5, stride=1, pad=1)
        out_2_h = get_out_size(out_1_h, kernel=3, stride=1, pad=1)
        out_3_h = get_out_size(out_2_h, kernel=3, stride=1, pad=1)

        out_1_w = get_out_size(INPUT_SIZE_WIDTH, kernel=5, stride=1, pad=1)
        out_2_w = get_out_size(out_1_w, kernel=3, stride=1, pad=1)
        out_3_w = get_out_size(out_2_w, kernel=3, stride=1, pad=1)

        self.fc = torch.nn.Linear(in_features=12 * out_3_h * out_3_w,
                                  out_features=n_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.encoder.forward(x)
        out_flat = out.view(batch_size, -1)
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
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
                metrics_strs.append(f'{key}: {round(value, 4)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()
