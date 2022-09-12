import os
import re
from glob import glob

from torch import nn
from tqdm import tqdm
import pandas as pd
from PIL import Image

from models import EffNet
from datasets import USDataset

from endaaman.torch import Trainer

def binary_acc_fn(outputs, labels):
    y_true = labels.flatten().detach().numpy() > 0.5
    y_pred = outputs.flatten().detach().numpy() > 0.5
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def binary_auc_fn(outputs, labels):
    y_true = labels.flatten().detach().numpy()
    y_pred = outputs.flatten().detach().numpy()
    return metrics.roc_auc_score(y_true, y_pred)


class C(Trainer):
    def run_train(self):
        train_loader = self.as_loader(USDataset(
            test=False,
        ))

        model = EffNet('b0')
        criterion = nn.BCELoss()

        def eval_fn(inputs, labels):
            outputs = model(inputs.to(self.device))
            loss = criterion(outputs, labels.to(self.device))
            return loss, outputs

        self.train_model(
            'b0',
            model,
            train_loader,
            None, # test_loader,
            eval_fn, {
                'acc': binary_acc_fn,
            }, {
                'auc': binary_auc_fn,
            })

if __name__ == '__main__':
    c = C()
    c.run()
