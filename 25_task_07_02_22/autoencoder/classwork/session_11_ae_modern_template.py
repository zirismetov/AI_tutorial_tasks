import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 5)

import torch.utils.data
import scipy.ndimage
import torch.nn.functional as F

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MAX_LEN = 500  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


# Link to dataset if download not working
# http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip to ../data/EMNIST/raw/emnist.zip

class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.data = torchvision.datasets.EMNIST(
            root='../data',
            train=is_train,
            split='byclass',
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def normalize(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_min == x_max or x_max == 0:
            return x
        return (x - x_min) / (x_max - x_min)

    def __getitem__(self, idx):
        pil_x, label_idx = self.data[idx]
        np_x = np.array(pil_x)  # (28, 28)
        np_x = np.expand_dims(np_x, axis=0)  # (C, W, H)
        np_x = self.normalize(np_x)
        x = torch.FloatTensor(np_x)

        return x


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetEMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetEMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)


class EncoderBlock(torch.nn.Module):
    def __init__(self, num_groups, in_channels, out_channels, is_down_sample):
        super().__init__()
        encoder = [
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        ]
        if is_down_sample:
            encoder.append(torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0))
        self.encoder = torch.nn.Sequential(*encoder).to(DEVICE)

    def forward(self, x):
        return self.encoder.forward(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, num_groups, in_channels, out_channels, is_upsample):
        super().__init__()
        Decoder = [
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),

            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            torch.nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels),
        ]
        if is_upsample:
            Decoder.append(torch.nn.UpsamplingBilinear2d(scale_factor=2))

        self.decoder = torch.nn.Sequential(*Decoder).to(DEVICE)

    def forward(self, x):
        return self.decoder.forward(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            EncoderBlock(in_channels=1,
                         out_channels=4,
                         num_groups=2,
                         is_down_sample=True),

            EncoderBlock(in_channels=4,
                         out_channels=8,
                         num_groups=2,
                         is_down_sample=True),

            EncoderBlock(in_channels=8,
                         out_channels=16,
                         num_groups=4,
                         is_down_sample=True),

            EncoderBlock(in_channels=16,
                         out_channels=32,
                         num_groups=4,
                         is_down_sample=False),

            torch.nn.Tanh()

        ).to(DEVICE)

        self.decoder = torch.nn.Sequential(
            DecoderBlock(in_channels=32, out_channels=16, num_groups=4, is_upsample=True),
            DecoderBlock(in_channels=16, out_channels=8, num_groups=4, is_upsample=True),
            DecoderBlock(in_channels=8, out_channels=4, num_groups=2, is_upsample=True),
            DecoderBlock(in_channels=4, out_channels=4, num_groups=2, is_upsample=True),
            DecoderBlock(in_channels=4, out_channels=4, num_groups=2, is_upsample=True),

            torch.nn.AdaptiveAvgPool2d(output_size=(28, 28)),
            DecoderBlock(in_channels=4, out_channels=1, num_groups=2, is_upsample=False),
            torch.nn.Sigmoid()

        ).to(DEVICE)

    def forward(self, x):
        z = self.encoder.forward(x)
        z = z.view(-1, 32)
        z_dec = z.view(-1, 32, 1, 1)

        y_prim = self.decoder.forward(z_dec)
        return y_prim, z


model = AutoEncoder()
# dummy = torch.randn((BATCH_SIZE, 1, 28, 28))
# y_prim, z = model.forward(dummy)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss'
    ]:
        metrics[f'{stage}_{metric}'] = []
best_loss = float('Inf')
for epoch in range(1, 100):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x in data_loader:
            if data_loader == data_loader_train:
                x = F.dropout(x, p=0.5)

            x = x.to(DEVICE)
            model = model.to(DEVICE)

            y_prim, z = model.forward(x)
            loss = torch.mean((x - y_prim)**2)

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()

            if data_loader == data_loader_test:
                loss_scalar = metrics_epoch[f'{stage}_loss'][-1]
                if loss_scalar < best_loss:
                    best_loss = loss_scalar
                    torch.save(model.cpu().state_dict(),
                               './best-model.pt')

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')
    if (epoch+1) % 10 == 0 :
        plt.subplot(121)  # row col idx
        plts = []
        c = 0
        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        for i, j in enumerate([4, 5, 6, 16, 17, 18]):
            plt.subplot(4, 6, j)
            plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))

            plt.subplot(4, 6, j + 6)
            plt.imshow(np_y_prim[i][0].T, cmap=plt.get_cmap('Greys'))

        plt.tight_layout(pad=0.5)
        plt.show()
