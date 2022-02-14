import argparse
import copy
import json
import math
import os
import pathlib
import random

import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy import spatial

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
import torch.utils.data
import torch.distributions

import matplotlib.pyplot as plt
from torchvision import transforms


BATCH_SIZE = 36

dataset = torchvision.datasets.EMNIST(
    root='/data',
    split='bymerge',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# examine samples to select idxes you would like to use


idx = 0
for x, y_idx in data_loader:
    plt.rcParams["figure.figsize"] = (int(BATCH_SIZE/4), int(BATCH_SIZE/4))
    plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
    for i in range(BATCH_SIZE):
        plt.subplot(plt_rows, plt_rows, i + 1)
        plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))
        plt.title(f"idx: {idx}")
        idx += 1
        plt.tight_layout(pad=0.5)
    plt.show()
    break
    # if input('inspect more samples? (y/n)') == 'n':
    #     break


class VAE2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=4, out_channels=8, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=8, out_channels=16, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=16, num_groups=8),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=16, out_channels=32, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=32, num_groups=8)
        )

        self.encoder_mu = torch.nn.Linear(in_features=32, out_features=32)
        self.encoder_sigma = torch.nn.Linear(in_features=32, out_features=32)

        self.decoder = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=32, out_channels=16, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=16, num_groups=8),


            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=16, out_channels=8, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),

            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=8, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),

            torch.nn.AdaptiveAvgPool2d(output_size=(28, 28)),
            torch.nn.Conv2d(in_channels=4, out_channels=1, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(normalized_shape=[1, 28, 28]),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)

        out_flat = out.view(x.size(0), -1)

        z_sigma = self.encoder_sigma.forward(out_flat)
        z_mu = self.encoder_mu.forward(out_flat)

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size())
        z = z_mu + eps * z_sigma

        z_2d = z.view(x.size(0), -1, 1, 1)
        y_prim = self.decoder(z_2d)
        return y_prim, z, z_sigma, z_mu

    def encode_z(self, x):
        out = self.encoder.forward(x)

        out_flat = out.view(x.size(0), -1)

        z_sigma = self.encoder_sigma.forward(out_flat)
        z_mu = self.encoder_mu.forward(out_flat)

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size())
        z = z_mu + eps * z_sigma

        return z

    def decode_z(self, z):
        z_2d = z.view(z.size(0), -1, 1, 1)
        y_prim = self.decoder(z_2d)
        return y_prim


model = VAE2()
model.load_state_dict(torch.load('mnist_mnist_bce_low_beta_fixed_log_epsilon-17-run-25.pt', map_location='cpu'))
model.eval()
torch.set_grad_enabled(False)


IDEXES_TO_ENCODE = [
    102, 80, 43, 23, 11, 86 # all these are same type of letter
]

x_to_encode = []
for idx in IDEXES_TO_ENCODE:
    x_to_encode.append(dataset[idx][0])

plt_rows = int(np.ceil(np.sqrt(len(x_to_encode))))
for i in range(len(x_to_encode)):
    plt.subplot(plt_rows, plt_rows, i + 1)
    x = x_to_encode[i]
    plt.imshow(x[0].T, cmap=plt.get_cmap('Greys'))
    plt.title(f"idx: {IDEXES_TO_ENCODE[i]}")
    plt.tight_layout(pad=0.5)
plt.show()

x_tensor = torch.stack(x_to_encode)
zs = model.encode_z(x_tensor)

z_mu = torch.mean(zs, dim=0)
z_sigma = torch.std(zs, dim=0)

# sample new letters
z_generated = []
dist = torch.distributions.Normal(z_mu, z_sigma)
for i in range(BATCH_SIZE):
    if i == 0:
        z = z_mu
    else:
        z = dist.sample()
    z_generated.append(z)
z = torch.stack(z_generated)
x_generated = model.decode_z(z)

plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
for i in range(BATCH_SIZE):
    plt.subplot(plt_rows, plt_rows, i + 1)
    plt.imshow(x_generated[i][0].T, cmap=plt.get_cmap('Greys'))
    if i == 0:
        plt.title(f"mean")
    else:
        plt.title(f"generated")
    plt.tight_layout(pad=0.5)
plt.show()