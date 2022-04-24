import time
from copy import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import ResNet

plt.style.use('dark_background')

import torchvision

from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

# pip install tensorflow
# pip install tensorboardX
# pip install argparse
# pip install tqdm

from torchvision.datasets import FashionMNIST
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-run_name', default=f'run', type=str)
parser.add_argument('-sequence_name', default=f'seq', type=str)
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

MAX_LEN = 0
# args.device = 'cpu'
cuda_available = torch.cuda.is_available()

if not cuda_available:
    args.device = 'cpu'
    MAX_LEN = 200
if args.device == 'cpu':
    cuda_available = False
    MAX_LEN = 200

class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data',
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x, dtype=np.float32) / 255.0
        np_x = np.expand_dims(np_x, axis=0)
        np_y = np.array([y_idx])
        return np_x, np_y


dataset_train = DatasetFashionMNIST(is_train=True)
dataset_test = DatasetFashionMNIST(is_train=False)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=args.batch_size,
    num_workers=8 if cuda_available else 0,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=args.batch_size,
    num_workers=8 if cuda_available else 0,
    shuffle=False
)


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.resnet_trained: ResNet = torchvision.models.resnet18(pretrained=True)
        conv_1_pretrained = self.resnet_trained.conv1  # (out_planes, 3 ,7, 7)
        self.resnet_trained.conv1 = torch.nn.Conv2d(in_channels=1,
                                                    out_channels=conv_1_pretrained.out_channels,
                                                    kernel_size=conv_1_pretrained.kernel_size,
                                                    padding=conv_1_pretrained.padding,
                                                    stride=conv_1_pretrained.stride,
                                                    bias=conv_1_pretrained.bias)
        self.resnet_trained.conv1.weight.data = torch.mean(
            conv_1_pretrained.weight.data, dim=1, keepdim=True)  # (out_planes, 1 ,7, 7) grey_scaled
        # transfer learning (1000 -> 10)
        self.resnet_trained.fc = torch.nn.Linear(in_features=self.resnet_trained.fc.in_features,
                                                 out_features=10)

    def forward(self, x):
        return self.resnet_trained.forward(x) # ResNet -> NO SOFTMAX AT THE END


model = Model(args).to(args.device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate
)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

loss_fn = torch.nn.CrossEntropyLoss() # if classes is not balanced in dataset weight is needed ( weights only for train set)


class TensorBoardSummaryWriter(SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=10, filename_suffix='',
                 write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue,
                         flush_secs, filename_suffix, write_to_disk,
                         log_dir, **kwargs)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        hparam_dict = copy(hparam_dict)
        for key in list(hparam_dict.keys()):
            if type(hparam_dict[key]) not in [float, int, str]:
                del hparam_dict[key]
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)


summary_writer = TensorBoardSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}_{int(time.time())}'
)

for epoch in range(1, args.epochs + 1):

    metrics = {}
    for dataloader in [data_loader_train, data_loader_test]:

        class_count = len(FashionMNIST.classes)
        conf_matrix = np.zeros((class_count, class_count))

        if dataloader == data_loader_train:
            mode = 'train'
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            mode = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)

        metrics[f'{mode}_loss'] = []
        metrics[f'{mode}_acc'] = []

        logits_with_labels_and_images = []

        for x, y in tqdm(dataloader, desc=mode):
            x = x.to(args.device)
            y = y.to(args.device).squeeze()
            y = y.type(torch.LongTensor)

            y_prim = model.forward(x)
            loss = loss_fn.forward(y_prim, y)
            metrics[f'{mode}_loss'].append(loss.cpu().item())

            if mode == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_idx = y.cpu().data.numpy()
            np_y_prim = y_prim.cpu().data.numpy()
            np_y_prim_idx = np_y_prim.argmax(axis=-1)
            acc = np.mean((np_y_idx == np_y_prim_idx) * 1.0)
            metrics[f'{mode}_acc'].append(acc)
            for idx in range(len(np_y_idx)):
                conf_matrix[np_y_prim_idx[idx], np_y_idx[idx]] += 1
                if len(logits_with_labels_and_images) < 1000:
                    logits_with_labels_and_images.append((
                        np_y_prim[idx],
                        FashionMNIST.classes[np_y_idx[idx]],
                        x.data.cpu().numpy()[idx]
                    ))


        fig = plt.figure()
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        plt.xticks(np.arange(class_count), FashionMNIST.classes, rotation=45)
        plt.yticks(np.arange(class_count), FashionMNIST.classes)
        for x in range(class_count):
            for y in range(class_count):
                perc = round(100 * conf_matrix[x, y] / np.sum(conf_matrix[x]), 1)
                plt.annotate(
                    str(perc),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor=(1., 1., 1., 0.),
                    color='black' if perc < 50 else 'white',
                    fontsize=7
                )
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.tight_layout(pad=0)

        summary_writer.add_figure(
            figure=fig,
            tag=f'{mode}_conf_matrix',
            global_step=epoch,

        )
        embs, labels, imgs = list(zip(*logits_with_labels_and_images))
        summary_writer.add_embedding(
            mat=np.array(embs),
            metadata=list(labels),
            label_img=np.array(imgs),
            tag= f'{mode}_logits',
            global_step=epoch
        )

    metrics_mean = {}
    for key in metrics:
        mean_value = np.mean(metrics[key])
        metrics_mean[key] = mean_value
        summary_writer.add_scalar( #ADD_SCALAR is for plots
            scalar_value=mean_value,
            tag = key,
            global_step=epoch
        )
    summary_writer.add_hparams(
         hparam_dict=args.__dict__,
         metric_dict=metrics_mean,
         name=args.run_name,
         global_step=epoch
    )
    summary_writer.flush()