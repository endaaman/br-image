import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics

from models import create_model
from datasets import USDataset

from endaaman.torch import Trainer

def binary_acc_fn(outputs, labels):
    y_true = labels.cpu().flatten().detach().numpy() > 0.5
    y_pred = outputs.cpu().flatten().detach().numpy() > 0.5
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def binary_auc_fn(outputs, labels):
    y_true = labels.cpu().flatten().detach().numpy()
    y_pred = outputs.cpu().flatten().detach().numpy()
    return metrics.roc_auc_score(y_true, y_pred)

available_models = [f'eff_b{i}' for i in range(6)] + ['vgg_{i}_bn' for i in [11, 13, 16, 19]]

class C(Trainer):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default=available_models[0])

    def arg_train(self, parser):
        parser.add_argument('--no-flip', action='store_true')
        parser.add_argument('--rotate', type=int, default=5)
        parser.add_argument('--shrink', type=float, default=0.2)
        parser.add_argument('--norm', choices=['l1', 'l2'])
        parser.add_argument('--alpha', type=float, default=0.01)

    def run_train(self):
        train_loader, test_loader = [self.as_loader(USDataset(
            test=t,
            a_flip=not self.args.no_flip,
            a_shrink=self.args.shrink,
            a_rotate=self.args.rotate,
        )) for t in [True, False]]

        model = create_model(self.args.model).to(self.device)
        criterion = nn.BCELoss()

        def eval_fn(inputs, labels):
            outputs = model(inputs.to(self.device))
            loss = criterion(outputs, labels.to(self.device))

            if self.args.norm in ['l1', 'l2']:
                l = torch.tensor(0., requires_grad=True)
                if self.args.norm == 'l1':
                    for w in model.parameters():
                        l = l + torch.norm(w, 1)
                elif self.args.norm == 'l2':
                    for w in model.parameters():
                        l = l + torch.norm(w)**2
                loss += l * self.args.alpha

            return loss, outputs

        self.train_model(
            self.args.model,
            model,
            train_loader,
            test_loader,
            eval_fn, {
                'acc': binary_acc_fn,
            }, {
                'auc': binary_auc_fn,
            })

if __name__ == '__main__':
    c = C({
        'epoch': 50,
        'lr': 0.001,
        'batch_size': 128,
    })
    c.run()
