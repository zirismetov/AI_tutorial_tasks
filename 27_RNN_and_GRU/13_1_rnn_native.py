import json
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data

# pip install nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# nltk.download('punkt')

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3

RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 2
RNN_DROPOUT = 0.3

run_path = ''

DEVICE = 'cpu'
# if torch.cuda.is_available():
#     DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if DEVICE == 'cuda':
    MAX_LEN = 10000

PATH_DATA = '../data'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)


def Freedman_Diaconis(x):
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self):
        if not os.path.exists(f'{PATH_DATA}/quotes.json'):
            download_url_to_file(
                'https://www.yellowrobot.xyz/share/recipes_raw_nosource_epi.json',
                f'{PATH_DATA}/recipes_raw_nosource_epi.json',
                progress=True
            )
        with open(f'{PATH_DATA}/quotes.json', encoding="utf8") as fp:
            data_json = json.load(fp)

        self.sentences = []
        self.lengths = []
        self.words_to_idxes = {}
        self.words_counts = {}
        self.idxes_to_words = {}

        for each_quote in data_json:
            str_quote = each_quote['Quote']
            word_list = []
            remove_char = ['/', "'", '"', "\\", ',', ".", "!", "?", "&", "*", '-', ":"]
            word_sep = str_quote.split()
            for word in word_sep:
                for letter in word:
                    if not letter in remove_char:
                        word_list.append(letter)
                    if letter in [',', ".", "!", "?"]:
                        word_list.append(" ")
                if not word == word_sep[-1]:
                    word_list.append(" ")
            str_quote = ''.join(word_list)
            sentences = sent_tokenize(str_quote)
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if len(words) > MAX_SENTENCE_LEN:
                    words = words[:MAX_SENTENCE_LEN]
                if len(words) < MIN_SENTENCE_LEN:
                    continue
                sentence_tokens = []
                for word in words:
                    if word not in self.words_to_idxes:
                        self.words_to_idxes[word] = len(self.words_to_idxes)
                        self.idxes_to_words[self.words_to_idxes[word]] = word
                        self.words_counts[word] = 0
                    self.words_counts[word] += 1
                    sentence_tokens.append(self.words_to_idxes[word])
                self.sentences.append(sentence_tokens)
                self.lengths.append(len(sentence_tokens))
            if MAX_LEN is not None and len(self.sentences) > MAX_LEN:
                break

        self.max_length = np.max(self.lengths) + 1

        self.end_token = '[END]'
        self.words_to_idxes[self.end_token] = len(self.words_to_idxes)
        self.idxes_to_words[self.words_to_idxes[self.end_token]] = self.end_token
        self.words_counts[self.end_token] = len(self.sentences)

        self.max_classes_tokens = len(self.words_to_idxes)

        word_counts = np.array(list(self.words_counts.values()))
        self.weights = (1.0 / word_counts) * np.sum(word_counts) * 0.5

        print(f'self.sentences: {len(self.sentences)}')
        print(f'self.max_length: {self.max_length}')
        print(f'self.max_classes_tokens: {self.max_classes_tokens}')

        print('Example sentences:')
        samples = np.random.choice(self.sentences, 5)
        for each in samples:
            print(' '.join([self.idxes_to_words[it] for it in each]))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        np_x_idxes = np.array(self.sentences[idx] + [self.words_to_idxes[self.end_token]])
        np_x_padded = np.zeros((self.max_length, self.max_classes_tokens))
        np_x_padded[np.arange(len(np_x_idxes)), np_x_idxes] = 1.0

        np_y_padded = np.roll(np_x_padded, shift=-1, axis=0)
        np_length = self.lengths[idx]

        return np_x_padded, np_y_padded, np_length


torch.manual_seed(0)
dataset_full = DatasetCustom()
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full) * 0.8), len(dataset_full) - int(len(dataset_full) * 0.8)])
torch.seed()

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


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        stdv = 1 / math.sqrt(hidden_size)
        self.W_h = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )
        self.W_x = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )
        self.W_y = torch.nn.Parameter(
            torch.FloatTensor(input_size, hidden_size).uniform_(-stdv, stdv)
        )

        self.bias_h = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).zero_()
        )
        self.bias_y = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).zero_()
        )

    def forward(self, x: PackedSequence, hidden=None):
        h_out = []
        x_unpack, lengths = pad_packed_sequence(x, batch_first=True)
        batch_size = x_unpack.size(0)
        if hidden is None:
            hidden = torch.FloatTensor(batch_size, self.hidden_size).zero_().to(DEVICE)

        x_seq = x_unpack.permute(1, 0, 2)

        for x_t in x_seq:
            # if model.training:
            #     x_t = torch.nn.functional.dropout(x_t, p=0.5)
            hidden = torch.tanh(
                hidden @ self.W_h +
                x_t @ self.W_x +
                self.bias_h
            )
            Y = hidden @ self.W_y + self.bias_y
            h_out.append(Y)

        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2)

        t_h_packed = pack_padded_sequence(t_h_out, lengths, batch_first=True)
        return t_h_packed


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=dataset_full.max_classes_tokens,
            embedding_dim=RNN_HIDDEN_SIZE
        )
        layers = []
        for _ in range(RNN_LAYERS):
            layers.append(RNNCell(
                input_size=RNN_HIDDEN_SIZE,
                hidden_size=RNN_HIDDEN_SIZE
            ))
        self.rnn = torch.nn.Sequential(*layers)

    def forward(self, x: PackedSequence, hidden=None):

        embs = self.embedding.forward(x.data.argmax(dim=1))
        embs_seq = PackedSequence(
            data=embs,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        hidden = self.rnn.forward(embs_seq)
        y_prim_logits = hidden.data @ self.embedding.weight.t()
        y_prim = torch.softmax(y_prim_logits, dim=1)
        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )
        return y_prim_packed, hidden


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
epoch = 0
acc = 0
while True:
    # for epoch in range(1, EPOCHS + 1):
    epoch += 1
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
            x_packed = pack_padded_sequence(x, lengths, batch_first=True)
            y_packed = pack_padded_sequence(y, lengths, batch_first=True)

            y_prim_packed, _ = model.forward(x_packed)

            weights = torch.from_numpy(dataset_full.weights[torch.argmax(y_packed.data, dim=1).cpu().numpy()])
            weights = weights.unsqueeze(dim=1).to(DEVICE)
            loss = -torch.mean(weights * y_packed.data * torch.log(y_prim_packed.data + 1e-8))

            metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

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
                metrics_strs.append(f'{key}: {round(value, 2)}')
        acc = metrics['test_acc']
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/model-{epoch}.pt')
        model = model.to(DEVICE)

    print('Examples:')
    y_prim_unpacked, lengths_unpacked = pad_packed_sequence(y_prim_packed.cpu(), batch_first=True)
    y_prim_unpacked = y_prim_unpacked[:5]  # 5 examples
    for idx, each in enumerate(y_prim_unpacked):
        length = lengths_unpacked[idx]

        y_prim_idxes = np.argmax(each[:length].data.numpy(), axis=1).tolist()
        x_idxes = np.argmax(x[idx, :length].cpu().data.numpy(), axis=1).tolist()
        y_prim_idxes = [x_idxes[0]] + y_prim_idxes
        print('x     : ' + ' '.join([dataset_full.idxes_to_words[it] for it in x_idxes]))
        print('y_prim: ' + ' '.join([dataset_full.idxes_to_words[it] for it in y_prim_idxes]))
        print('')

    plt.figure(figsize=(12, 5))
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig(f'./results/epoch-{epoch}.png')
    plt.show()
