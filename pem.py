import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
import numpy as np
from pydantic import Field
import torch
from torch import nn
from torch import optim
from timm.scheduler.cosine_lr import CosineLRScheduler

from endaaman.cli import BaseCLI
from endaaman.ml import define_torch_args
from endaaman.trainer import Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model
from datasets import PEMDataset


class MyTrainer(Trainer):
    def prepare(self, extra):
        extra = extra or {}
        self.cosine = extra.pop('cosine', -1)
        assert len(extra) == 0

        # self.criterion = FocalBCELoss(gamma=4.0)
        self.criterion = nn.BCELoss()
        model = create_model(self.model_name)
        return model.to(self.device)

    def create_scheduler(self, total_epoch):
        if self.cosine > 0:
            warmup = 5
            return CosineLRScheduler(
                self.optimizer,
                warmup_t=warmup, t_initial=total_epoch-warmup,
                warmup_lr_init=self.lr/2, lr_min=self.lr/self.cosine,
                warmup_prefix=True)
        return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)


    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        if isinstance(self.scheduler, CosineLRScheduler):
            self.scheduler.step(self.current_epoch)
            return
        super().step(train_loss)

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

DefaultArgs = define_torch_args(
    epoch=30,
    lr=0.0001,
    batch_size=16,
)


class CLI(BaseCLI):
    class CommonArgs(DefaultArgs):
        model_name:str = Field('tf_efficientnetv2_b0', cli=('--model', ))

    class StartArgs(CommonArgs):
        size:int = 512
        mode:str = 'pe'
        cosine:int = -1
        split:str = '0.25'

    def run_start(self, a:StartArgs):
        try:
            split = float(a.split)
        except ValueError as _:
            split = a.split

        loaders = [a.as_loader(PEMDataset(
            size=a.size,
            target=t,
            mode=a.mode,
            split=split,
        )) for t in ['train', 'test']]

        trainer = a.create_trainer(
            TrainerClass=MyTrainer,
            model_name=a.model_name,
            loaders=loaders,
            experiment_name=self.a.experiment_name,
            extra=dict(
                cosine=self.a.cosine,
            )
        )

        trainer.start(a.epoch)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
