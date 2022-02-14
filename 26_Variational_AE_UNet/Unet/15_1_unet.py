import torch
import numpy as np
import matplotlib
import torchvision
import math
import os
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 5)
import torch.utils.data

USE_CUDA = torch.cuda.is_available()


class DatasetLungs(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.class_count = 3
        self.path = './data_lung'
        x = np.load(os.path.join(self.path, 'x.npy'))
        y = np.load(os.path.join(self.path, 'y.npy'))
        self.data = []
        for idx in range(y.shape[0]):
            self.data.append((x[idx], y[idx]))
        if is_train:
            self.data = self.data[:int(len(self) * 0.8)]
        else:
            self.data = self.data[int(len(self) * 0.8):]

    def normalize(self, data):
        data_max = data.max()
        data_min = data.min()
        if data_min != data_max:
            data = ((data - data_min) / (data_max - data_min))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = self.normalize(x)
        x = torch.FloatTensor(x)
        y = self.normalize(y)
        y = torch.FloatTensor(y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetLungs(is_train=True),
    batch_size=16,
    shuffle=True,
    drop_last=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetLungs(is_train=False),
    batch_size=16,
    shuffle=False,
    drop_last=True
)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels / 2))
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels / 2))
        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.gn1(out)
        out = self.conv2(out)
        if self.is_bottleneck:
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        out = self.gn2(out)
        return out


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_down_1 = ResBlock(1, 8)
        self.conv_down_2 = ResBlock(8, 16)
        self.conv_down_3 = ResBlock(16, 32)
        self.conv_down_4 = ResBlock(32, 64)

        self.conv_middle = ResBlock(64, 64)

        self.conv_up_4 = ResBlock(128, 32)
        self.conv_up_3 = ResBlock(64, 16)
        self.conv_up_2 = ResBlock(32, 8)
        self.conv_up_1 = ResBlock(16, 1)

        # self.conv_up_4 = ResBlock(64, 32)
        # self.conv_up_3 = ResBlock(32, 16)
        # self.conv_up_2 = ResBlock(16, 8)
        # self.conv_up_1 = ResBlock(8, 1)

    def forward(self, x):
        out1 = self.conv_down_1.forward(x)
        out2 = self.conv_down_2.forward(self.maxpool.forward(out1))
        out3 = self.conv_down_3.forward(self.maxpool.forward(out2))
        out4 = self.conv_down_4.forward(self.maxpool.forward(out3))
        out5 = self.conv_middle.forward(out4)
        out6 = self.conv_up_4.forward(torch.cat([out5, out4], dim=1))
        out7 = self.conv_up_3.forward(torch.cat([self.upsample.forward(out6), out3], dim=1))
        out8 = self.conv_up_2.forward(torch.cat([self.upsample.forward(out7), out2], dim=1))
        y_prim = torch.sigmoid(self.conv_up_1.forward(torch.cat([self.upsample.forward(out8), out1], dim=1)))
        return y_prim
    # def forward(self, x):
    #     out1 = self.conv_down_1.forward(x)
    #     out2 = self.conv_down_2.forward(self.maxpool.forward(out1))
    #     out3 = self.conv_down_3.forward(self.maxpool.forward(out2))
    #     out4 = self.conv_down_4.forward(self.maxpool.forward(out3))
    #     out5 = self.conv_middle.forward(out4)
    #     out6 = self.conv_up_4.forward(out5 + out4)
    #     out7 = self.conv_up_3.forward(self.upsample.forward(out6) + out3)
    #     out8 = self.conv_up_2.forward(self.upsample.forward(out7) + out2)
    #     y_prim = torch.sigmoid(self.conv_up_1.forward(self.upsample.forward(out8) + out1))
    #     return y_prim


model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if USE_CUDA:
    model = model.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'IoU'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}
        stage = 'train'
        torch.set_grad_enabled(True)
        model = model.train()
        if data_loader == data_loader_test:
            stage = 'test'
            torch.set_grad_enabled(False)
            model = model.eval()

        for x, y in tqdm(data_loader):

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            # Implement BCE loss
            y_prim = model.forward(x)
            BCE_loss = -torch.mean(y * torch.log(y_prim + 1e-8) + (1.0 - y) * torch.log((1.0 - y_prim) + 1e-8))
            Dice_loss = 2 * torch.sum(y * y_prim) / (torch.sum(y) + torch.sum(y_prim) + 1e-8)
            loss = BCE_loss + (1 - Dice_loss)
            metrics_epoch[f'{stage}_loss'].append(loss.item())
            iou_denominator = torch.sum((y * y_prim) +
                                        (1 - y) * y_prim +
                                        y * (1 - y_prim))

            iou_aka_jaccob_index = (torch.sum(y * y_prim) + 1e-8) / (iou_denominator + 1e-8)
            metrics_epoch[f'{stage}_IoU'].append(iou_aka_jaccob_index.item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.cpu()
            y_prim = y_prim.cpu()
            x = x.cpu()
            y = y.cpu()

            np_x = x.data.numpy()
            np_y_prim = y_prim.data.numpy()
            np_y = y.data.numpy()

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.subplot(121)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18]):
        plt.subplot(4, 6, j)
        plt.title('y')
        plt.imshow(np_x[i][0], cmap='Greys', interpolation=None)
        plt.imshow(np_y[i][0], cmap='Reds', alpha=0.5, interpolation=None)
        plt.subplot(4, 6, j + 6)
        plt.title('y_prim')
        plt.imshow(np_x[i][0], cmap='Greys', interpolation=None)
        plt.imshow(np.where(np_y_prim[i][0] > 0.8, np_y_prim[i][0], 0), cmap='Reds', alpha=0.5, interpolation=None)

    plt.tight_layout(pad=0.5)
    plt.show()
