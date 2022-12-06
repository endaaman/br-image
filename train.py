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
from endaaman.torch import TrainCommander
from endaaman.trainer import Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model, available_models
from datasets import USDataset


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

    def eval(self, inputs, labels):
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs

    def get_metrics(self):
        return {
            'batch': {
                'acc': BinaryAccuracy(),
                'recall': BinaryRecall(),
                'spec': BinarySpecificity()
            },
            'epoch': {
                'auc': BinaryAUC(),
            },
        }


class CMD(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default='eff_b0')

    def arg_start(self, parser):
        parser.add_argument('--size', type=int, default=512)
        parser.add_argument('--short', action='store_true')

    def run_start(self):
        model = create_model(self.args.model, 1)

        loaders = [self.as_loader(USDataset(
            size=self.args.size,
            target=t,
            len_scale=0.02 if self.args.short else 1,
        )) for t in ['train', 'test']]

        trainer = T(
            name=self.args.model,
            model=model,
            loaders=loaders,
            device=self.device,
            save_period=self.args.save_period,
            suffix=self.args.suffix,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 50,
        'lr': 0.0001,
        'batch_size': 16,
    })
    cmd.run()
