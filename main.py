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
from endaaman.torch import Trainer
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

class C(Trainer):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default='eff_b0')

    def arg_train(self, parser):
        parser.add_argument('--norm', choices=['l1', 'l2'])
        parser.add_argument('--alpha', type=float, default=0.01)
        parser.add_argument('--size', type=int)
        parser.add_argument('--short', action='store_true')

    def run_train(self):
        model, size = create_model(self.args.model)
        model.to(self.device)

        train_loader, test_loader = [self.as_loader(USDataset(
            size=self.args.size or size,
            target=t,
            len_scale=0.02 if self.args.short else 2,
        )) for t in ['train', 'test']]

        criterion = FocalBCELoss(gamma=4.0)
        def scheduler_fn(optimizer):
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.99 ** x)

        def eval_fn(inputs, labels):
            outputs = model(inputs.to(self.device))
            loss = criterion(outputs, labels.to(self.device))
            return loss, outputs

        self.train_model(
            name=self.args.model,
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            eval_fn=eval_fn,
            # scheduler_fn=scheduler_fn,
            scheduler_fn=None,
            batch_metrics={
                'acc': BinaryAccuracy(),
                'recall': BinaryRecall(),
                'spec': BinarySpecificity()
            },
            epoch_metrics={
                'auc': BinaryAUC(),
            },
        )

if __name__ == '__main__':
    c = C({
        'epoch': 100,
        'lr': 0.0001,
        'batch_size': 64,
    })
    c.run()
