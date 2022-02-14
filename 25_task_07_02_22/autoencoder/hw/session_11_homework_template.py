import torch
import numpy as np
import torchvision.transforms as tf
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

model_path= './pre_trained_weights.pt'

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=5, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.Conv2d(4, 8, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(8, 8, kernel_size=7, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(8, 16, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(16, 16, kernel_size=4, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=32)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=4, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(8, 8, kernel_size=7, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(8, 4, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=5, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder.forward(x)
        out = self.decoder.forward(out)
        return out
# make model instance
model = Autoencoder()

# load model weights
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


def normalize(data):
    data_max = data.max()
    data_min = data.min()
    if data_min != data_max:
        data = ((data - data_min) / (data_max - data_min))
    return data

# tasks on full dataset
dataset =torchvision.datasets.EMNIST(
    root='C:/Users/Kekuzbek/PycharmProjects/AI_tutorial_tasks/data',
    split='bymerge',
    train=False,
    download=True,
    transform=tf.ToTensor()
)
x = np.expand_dims(dataset.data.numpy(), axis=1)
x = normalize(x)
x = torch.FloatTensor(x)

# fill with reference images
reference_images = torch.empty((len(dataset.classes), *x[0].size()))
samples_to_remove = []
for idx in range(len(dataset.classes)):
    sample_idx = np.where(dataset.targets == idx)[0][0]    # select one sample from class idx
    sample = x[sample_idx]  # select sample
    reference_images[idx] = sample
    samples_to_remove.append(sample_idx)

samples_to_remove = sorted(samples_to_remove, reverse=True)
for idx in samples_to_remove:
    if idx != 0:
        x = torch.cat((x[:,:idx], x[:,idx+1:]), axis=1)

# process reference images
with torch.no_grad():
    refence_embedding = model.encoder(reference_images)
    refence_embedding_flat = refence_embedding.view(refence_embedding.size(0), -1)
    reference_decoded = model.decoder(refence_embedding_flat.view(-1,32,1,1))

# torch.set_grad_enabled(False)

# a lot of first samples will be duplicates from reference images
for sample_number, sample in enumerate(x):
    sample = sample.view(1,1,28,28)
    sample_decoded = model.forward(sample)

    # calculate cosine distance for each reference image
    dist_distances = {}
    for idx in range(len(reference_decoded)):
        u = sample_decoded.detach().view(-1).numpy()
        v = reference_decoded[idx].detach().view(-1).numpy()
        denominator = (np.sqrt(np.sum((u**2))) * np.sqrt(np.sum((v**2))))
        distance = np.mean(1 - ((u @ v) / denominator ))
        # select only close to zero values
        if 0< distance < 1:
            dist_distances[idx] = distance

    if sample_number % 10 == 0:
        # get ids for 6 closet reference images
        needed_min_id = sorted(dist_distances.items(), key=lambda x: x[1])

        plt.subplot(121)
        plt.plot(dist_distances.values(), 'o')
        for idx, value in dist_distances.items():
            plt.annotate(idx, (idx, value))

        # show six closet reference images(bottom) and original sample (top)
        for i, j in enumerate([4, 5, 6, 16, 17, 18]):
            plt.subplot(4, 6, j)
            plt.imshow(sample_decoded.detach().numpy()[0].T, cmap=plt.get_cmap('Greys'))
            plt.subplot(4, 6, j + 6)
            plt.imshow(reference_decoded[needed_min_id[i][0]].detach().numpy().T, cmap=plt.get_cmap('Greys'))
            plt.ylabel(f'idx_{needed_min_id[i][0]}')

        plt.savefig(f'plots/plot_{sample_number}.png')
        plt.show()

