import os
import re
from glob import glob
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid, save_image
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# from gradcam.utils import visualize_cam
# from gradcam import GradCAM, GradCAMpp
from endaaman.ml import TorchCommander, pil_to_tensor, Predictor
from endaaman.metrics import MultiAccuracy
from endaaman.utils import with_wrote, get_images_from_dir_or_file

from models import create_model
from datasets import PEMDataset, MEAN, STD


class MyPredictor(Predictor):
    def prepare(self, **kwargs):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 48)
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        model = create_model(self.checkpoint.model_name).to(self.device)
        model.load_state_dict(self.checkpoint.model_state)
        model.eval()
        return model

    def eval(self, inputs):
        features, pp = self.model(inputs.to(self.device), activate=True, with_features=True)
        return list(zip(features, pp))

    def collate(self, pred, idx):
        return pred

    # def predict_image(self, image, grid_size=-1):
    #     if grid_size < 0:
    #         return self.predict_images([image], batch_size=1)[0]
    #     ii = grid_split(image, size=grid_size, flattern=True)
    #     preds = self.predict_images(ii, batch_size=self.batch_size)
    #     return torch.stack(preds).sum(dim=0) / len(preds)



class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)

    def pre_common(self):
        self.checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        # self.checkpoint = torch.load(self.args.checkpoint)
        self.predictor = MyPredictor(self.checkpoint, self.a.batch_size, self.device)

    def arg_dataset(self, parser):
        # parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--mode', default='pe')
        parser.add_argument('--out-dir', '-o', default='out')
        parser.add_argument('--split', default='0.25')

    def run_dataset(self):
        try:
            split = float(self.a.split)
        except ValueError as _:
            split = self.a.split

        dataset = PEMDataset(
            target='all',
            mode=self.a.mode,
            normalize=False,
            aug_mode='none',
            train_test=split,
        )

        results = self.predictor.predict([i.image for i in dataset.items])
        pp, features = zip(*results)

        # each items
        oo = []
        for (item, p) in zip(dataset.items, pp):
            pred = float(p.tolist()[0])
            oo.append({
                'name': item.name,
                'id': item.id,
                'test': int(item.test),
                'gt': int(item.diagnosis),
                'pred': pred,
            })
        df_all = pd.DataFrame(oo)

        dfs = {
            'test': df_all[df_all['test'] == 1],
            'train': df_all[df_all['test'] == 0],
            'all': df_all,
        }

        # metrics by conditions
        mm = OrderedDict()
        for t in ['train', 'test', 'all']:
            df = dfs[t]
            m = {}
            fpr, tpr, thresholds = metrics.roc_curve(df['gt'], df['pred'])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{t} auc={auc:.3f}')

            scoress = {
                # f1: point to maximize f1 score
                'f1': [metrics.f1_score(df['gt'], df['pred'] > t) for t in thresholds],
                # youden: tanget to 45degree line
                'youden': tpr - fpr,
                # top-left: nearest to top-left corner
                'top-left': -((- tpr + 1) ** 2 + fpr ** 2),
            }

            for name, scores in scoress.items():
                idx = np.argmax(scores)
                m[name] = {
                    'threshold': thresholds[idx],
                    'recall': tpr[idx],
                    'spec': 1 - fpr[idx],
                }

            mm[t] = pd.DataFrame(m)

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)

        d = os.path.join(self.args.out_dir, self.checkpoint.trainer_name)
        os.makedirs(d, exist_ok=True)

        plt.savefig(os.path.join(d, 'roc.png'))
        plt.close()

        dest_path = os.path.join(d, 'report.xlsx')
        with pd.ExcelWriter(dest_path) as w: # pylint: disable=abstract-class-instantiated
            df.to_excel(w, sheet_name='values', index=False)
            for t, m in mm.items():
                m.to_excel(w, sheet_name=f'{t} metrics')

        features_dir = os.path.join(d, 'features')
        os.makedirs(features_dir, exist_ok=True)
        for (item, f) in zip(dataset.items, features):
            path = os.path.join(features_dir, f'{item.name}')
            np.save(path, f.cpu().detach().numpy())

    def arg_predict(self, parser):
        parser.add_argument('--src', '-s', required=True)

    def run_predict(self):
        images, paths = get_images_from_dir_or_file(self.args.src, with_path=True)

        results = self.predict_images(images)

        for (path, result) in zip(paths, results):
            print(f'{path}: {result:.2f}')

if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
