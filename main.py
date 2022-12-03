import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
from endaaman.torch import TrainCommander, Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model
from datasets import USDataset

available_models = \
    [f'eff_b{i}' for i in range(6)] + \
    [f'vgg{i}' for i in [11, 13, 16, 19]] + \
    [f'vgg{i}_bn' for i in [11, 13, 16, 19]]


class FocalBCELoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, outputs, targets):
        bce = self.bceloss(outputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = (1-bce_exp)**self.gamma * bce
        return focal_loss.mean()

class T(Trainer):
    def prepare(self):
        self.criterion = FocalBCELoss(gamma=4.0)

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.99 ** x)

    def eval(self, inputs, labels, device):
        outputs = self.model(inputs.to(device))
        loss = self.criterion(outputs, labels.to(device))
        return loss, outputs

    def get_batch_metrics(self):
        return {
            'acc': BinaryAccuracy(),
            'recall': BinaryRecall(),
            'spec': BinarySpecificity()
        }

    def get_epoch_metrics(self):
        return {
            'auc': BinaryAUC(),
        }


class CMD(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default='eff_b0')

    def arg_train(self, parser):
        parser.add_argument('--norm', choices=['l1', 'l2'])
        parser.add_argument('--alpha', type=float, default=0.01)
        parser.add_argument('--size', type=int)
        parser.add_argument('--short', action='store_true')

    def run_train(self):
        model, size = create_model(self.args.model)

        loaders = [self.as_loader(USDataset(
            size=self.args.size or size,
            target=t,
            len_scale=0.02 if self.args.short else 1,
        )) for t in ['train', 'test']]

        trainer = T(
            name=self.args.model,
            model=model,
            loaders=loaders,
        )

        trainer.train(self.args.lr, self.args.epoch, device=self.device)

if __name__ == '__main__':
    cmd = CMD({
        'epoch': 50,
        'lr': 0.0001,
        'batch_size': 16,
    })
    cmd.run()
