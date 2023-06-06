import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
import numpy as np
from sklearn import metrics as skmetrics
from pydantic import Field
import torch
from torch import nn
from torch import optim
from timm.scheduler.cosine_lr import CosineLRScheduler

from endaaman.ml import BaseDLArgs, BaseMLCLI, BaseTrainer, BaseTrainerConfig
from endaaman.metrics import ROCMetrics

from models import TimmModel
from datasets import PEMDataset


class TrainerConfig(BaseTrainerConfig):
    model_name: str
    mode: str
    split: str|float
    size: int


class Trainer(BaseTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = TimmModel(self.config.model_name, 1)
        return model.to(self.device)

    def eval(self, inputs, gts):
        preds = self.model(inputs.to(self.device))
        loss = self.criterion(preds, gts.to(self.device))
        return loss, preds.detach().cpu()

    def visualize_roc(self, ax, train_preds, train_gts, val_preds, val_gts):
        for t, preds, gts in (('train', train_preds, train_gts), ('val', val_preds, val_gts)):
            fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
            auc = skmetrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{t} AUC:{auc:.3f}')
            if t == 'train':
                youden_index = np.argmax(tpr - fpr)
                threshold = thresholds[youden_index]
        ax.set_title(f'ROC (t={threshold:.2f})')
        ax.set_ylabel('Sensitivity')
        ax.set_xlabel('1 - Specificity')
        ax.legend(loc='lower right')

    def get_metrics(self):
        return {
            'auc_acc_recall_spec': ROCMetrics(keys=['auc', 'acc', 'recall', 'specificity'])
        }


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(BaseDLArgs):
        epoch:int = 30
        batch_size:int = Field(2, cli=('--batch-size', '-B', ))
        lr:float = 0.001
        model_name:str = Field('tf_efficientnet_b0', cli=('--model', ))

        size:int = 512
        mode:str = 'pe'
        split:str = '0.25'

    def run_train(self, a:TrainArgs):
        try:
            split = float(a.split)
        except ValueError as _:
            split = a.split

        config = TrainerConfig(
            lr=a.lr,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            model_name=a.model_name,
            mode=a.mode,
            split=a.split,
            size=a.size,
        )

        dss = [PEMDataset(
            size=a.size,
            target=t,
            mode=a.mode,
            split=split,
        ) for t in ['train', 'test']]

        trainer = Trainer(
            config=config,
            out_dir=f'out/classification/{a.mode}/{a.model_name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu,
            experiment_name=a.mode,
            main_metrics='auc',
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
