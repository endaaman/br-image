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
from endaaman.trainer import Trainer, TrainCommander
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_seg_model
from datasets import USDataset


class T(Trainer):
    def prepare(self, **kwargs):
        # self.criterion = FocalBCELoss(gamma=4.0)
        self.criterion = nn.BCELoss()

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
        outputs = outputs.flatten(start_dim=1)
        labels = labels.flatten(start_dim=1)
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs

    def get_metrics(self):
        return {
            'batch': { },
            'epoch': { },
        }


class CMD(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', default='seg_eff_b0')

    def arg_start(self, parser):
        parser.add_argument('--size', type=int, default=512)

    def run_start(self):
        model = create_seg_model(self.args.model)

        loaders = [self.as_loader(USDataset(
            size=self.args.size,
            target=t,
            mode='seg',
        )) for t in ['train', 'test']]

        trainer = self.create_trainer(
            T=T,
            name=self.a.model,
            model=model,
            loaders=loaders,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 100,
        'lr': 0.0001,
        'batch_size': 16,
    })
    cmd.run()
