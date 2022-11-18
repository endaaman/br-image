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

from models import create_model
from datasets import USDataset

def binary_acc_fn(outputs, labels):
    y_true = labels.cpu().flatten().detach().numpy() > 0.5
    y_pred = outputs.cpu().flatten().detach().numpy() > 0.5
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def binary_auc_fn(outputs, labels):
    y_true = labels.cpu().flatten().detach().numpy()
    y_pred = outputs.cpu().flatten().detach().numpy()
    return metrics.roc_auc_score(y_true, y_pred)

available_models = \
    [f'eff_b{i}' for i in range(6)] + \
    [f'vgg{i}' for i in [11, 13, 16, 19]] + \
    [f'vgg{i}_bn' for i in [11, 13, 16, 19]]

class C(Trainer):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default='eff_b0')

    def arg_train(self, parser):
        parser.add_argument('--norm', choices=['l1', 'l2'])
        parser.add_argument('--alpha', type=float, default=0.01)
        parser.add_argument('--size', type=int)

    def run_train(self):
        model, size = create_model(self.args.model)
        model.to(self.device)
        self.prepare(model)

        train_loader, test_loader = [self.as_loader(USDataset(
            size=self.args.size or size,
            target=t,
        )) for t in ['train', 'test']]

        criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 0.99 ** x)

        def eval_fn(inputs, labels):
            outputs = model(inputs.to(self.device))
            loss = criterion(outputs, labels.to(self.device))
            return loss, outputs

        self.train_model(
            name=self.args.model,
            train_loader=train_loader,
            val_loader=test_loader,
            eval_fn=eval_fn,
            batch_metrics={
                'acc': binary_acc_fn,
            },
            epoch_metrics={
                # 'auc': binary_auc_fn,
            },
            short=True,
        )

if __name__ == '__main__':
    c = C({
        'epoch': 50,
        'lr': 0.001,
        'batch_size': 128,
    })
    c.run()
