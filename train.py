import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
import numpy as np
import torch
from torch import nn
from torch import optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from endaaman.torch import TrainCommander
from endaaman.trainer import Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model
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

    def create_scheduler(self, lr):
        return CosineLRScheduler(
            self.optimizer, t_initial=100, lr_min=0.00001,
            warmup_t=10, warmup_lr_init=0.00005, warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

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
        parser.add_argument('--model', '-m', default='eff_v2_b0')

    def arg_start(self, parser):
        parser.add_argument('--size', type=int, default=512)
        parser.add_argument('--short', action='store_true')
        parser.add_argument('--mode', default='pem')

    def run_start(self):
        model = create_model(self.args.model, 1)

        loaders = [self.as_loader(USDataset(
            size=self.args.size,
            target=t,
            mode=self.args.mode,
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
        'epoch': 100,
        'lr': 0.0001,
        'batch_size': 16,
    })
    cmd.run()
