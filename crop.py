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
from endaaman.ml import TrainCommander
from endaaman.trainer import Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model
from datasets import CroppedDataset


class T(Trainer):
    def prepare(self, **kwargs):
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
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0')

    def arg_start(self, parser):
        parser.add_argument('--size', type=int, default=512)
        parser.add_argument('--short', action='store_true')

    def run_start(self):
        model = create_model(self.args.model)

        loaders = [self.as_loader(CroppedDataset(
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
        'epoch': 100,
        'lr': 0.0001,
        'batch_size': 16,
    })
    cmd.run()
