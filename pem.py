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
from endaaman.torch import TorchCommander
from endaaman.trainer import Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model
from datasets import PEMDataset


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

class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        self.t_warmup = kwargs.pop('lr_warmup', 5)
        self.lr_min = kwargs.pop('lr_min', self.lr/10)

        # self.criterion = FocalBCELoss(gamma=4.0)
        self.criterion = nn.BCELoss()
        model = create_model(self.model_name)
        return model.to(self.device)

    def create_scheduler(self, total_epoch):
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=self.t_warmup, t_initial=total_epoch,
            warmup_lr_init=self.lr/2, lr_min=self.lr_min,
            warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def eval(self, inputs, gts):
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, gts.to(self.device))
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


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0')

    def arg_start(self, parser):
        parser.add_argument('--size', type=int, default=512)
        parser.add_argument('--short', action='store_true')
        parser.add_argument('--mode', default='pe')
        parser.add_argument('--split', default='0.25')

    def run_start(self):
        try:
            split = float(self.a.split)
        except ValueError as _:
            split = self.a.split

        loaders = [self.as_loader(PEMDataset(
            size=self.args.size,
            target=t,
            mode=self.args.mode,
            train_test=split,
            len_scale=0.02 if self.args.short else 1,
        )) for t in ['train', 'test']]

        trainer = self.create_trainer(
            MyTrainer,
            model_name=self.a.model,
            loaders=loaders,
        )

        trainer.start(self.args.epoch)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 30,
        'lr': 0.0001,
        'batch_size': 16,
    })
    cmd.run()
