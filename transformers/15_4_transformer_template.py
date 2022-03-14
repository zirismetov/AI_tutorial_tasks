
import json
import os
import pdb
import pickle

import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data

# pip install nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 1e-3

HIDDEN_SIZE = 64
TRANSFORMER_LAYERS = 8
DROPOUT = 0.1
run_name = 'add_full'

TRANSFORMER_HEADS = 4

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if DEVICE == 'cuda':
    MAX_LEN = None

PATH_DATA = '../data'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)

class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self):
        self.sentences = []
        self.lengths = []
        self.words_to_idxes = {}
        self.words_counts = {}
        self.idxes_to_words = {}
        self.max_length = 0
        self.end_token = '[END]'
        self.max_classes_tokens = len(self.words_to_idxes)
        self.weights = 0

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.sentences)

    def __getitem__(self, idx):
        np_x_idxes = np.array(self.sentences[idx] + [self.words_to_idxes[self.end_token]])
        np_x_padded = np.zeros((self.max_length, self.max_classes_tokens))
        np_x_padded[np.arange(len(np_x_idxes)), np_x_idxes] = 1.0

        np_y_padded = np.roll(np_x_padded, shift=-1, axis=0)
        np_length = self.lengths[idx]

        return np_x_padded, np_y_padded, np_length

if not os.path.exists(f'{PATH_DATA}/dataset_full.pt'):
    download_url_to_file('http://www.yellowrobot.xyz/share/dataset_full.pt', f'{PATH_DATA}/dataset_full.pt', progress=True)

with open(f'{PATH_DATA}/dataset_full.pt', 'rb') as fp:
    dataset_full = pickle.load(fp)

torch.manual_seed(0)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])
torch.seed()

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, idxes):
        return self.pe[idxes, :]


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #TODO

    def forward(self, x, lengths, atten):

        #TODO
        y_prim = x
        return y_prim, lengths, atten


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #TODO

    def forward(self, x: PackedSequence):

        x_e = PackedSequence(
            data=self.embeddings.forward(x.data.argmax(dim=1)),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        x_e_unpacked, lengths = pad_packed_sequence(x_e, batch_first=True)

        #TODO
        y_prim = 0

        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        return y_prim_packed, atten

model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):


    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y, lengths in data_loader:

            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)

            idxes = torch.argsort(lengths, descending=True)
            lengths = lengths[idxes]
            max_len = int(lengths.max())
            x = x[idxes, :max_len]
            y = y[idxes, :max_len]
            if int(lengths.min()) == 0:
                pdb.set_trace()
            x_packed = pack_padded_sequence(x, lengths, batch_first=True)
            y_packed = pack_padded_sequence(y, lengths, batch_first=True)

            y_prim_packed, atten = model.forward(x_packed)

            weights = torch.from_numpy(dataset_full.weights[torch.argmax(y_packed.data, dim=1).cpu().numpy()])
            weights = weights.unsqueeze(dim=1).to(DEVICE)
            loss = -torch.mean(weights * y_packed.data * torch.log(y_prim_packed.data + 1e-8))

            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim_packed.data.cpu().data.numpy()
            np_y = y_packed.data.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 4)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/{run_name}-model-{epoch}.pt')
        model = model.to(DEVICE)

        imageio.imwrite(f'./results/{run_name}-epoch-{epoch}-atten-0.png', atten[0].cpu().data.numpy())
        imageio.imwrite(f'./results/{run_name}-epoch-{epoch}-atten-l.png', atten[-1].cpu().data.numpy())


    print('Examples:')
    y_prim_unpacked, lengths_unpacked = pad_packed_sequence(y_prim_packed.cpu(), batch_first=True)
    y_prim_unpacked = y_prim_unpacked[:5] # 5 examples
    for idx, each in enumerate(y_prim_unpacked):
        length = lengths_unpacked[idx]

        y_prim_idxes = np.argmax(each[:length].data.numpy(), axis=1).tolist()
        x_idxes = np.argmax(x[idx, :length].cpu().data.numpy(), axis=1).tolist()
        y_prim_idxes = [x_idxes[0]] + y_prim_idxes
        print('x     : ' +' '.join([dataset_full.idxes_to_words[it] for it in x_idxes]))
        print('y_prim: ' +' '.join([dataset_full.idxes_to_words[it] for it in y_prim_idxes]))
        print('')

    plt.figure(figsize=(12,5))
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig(f'./results/{run_name}-epoch-{epoch}.png')
    plt.show()