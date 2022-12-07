import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid, save_image
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics

# from gradcam.utils import visualize_cam
# from gradcam import GradCAM, GradCAMpp
# from endaaman.torch import TorchCommander, pil_to_tensor
# from endaaman.metrics import MultiAccuracy

from models import create_model, available_models
from datasets import USDataset, MEAN, STD



class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)

    def pre_common(self):
        self.checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        # self.checkpoint = torch.load(self.args.checkpoint)
        self.model_name = self.checkpoint.name
        self.model = create_model(self.model_name, 1).to(self.device)
        self.model.load_state_dict(self.checkpoint.model_state)
        self.model.eval()

        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 24)

    def predict_images(self, images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        outputs = []
        for image in tqdm(images):
            t = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                o = self.model(t).detach().cpu()
            outputs += o
        return outputs


    def arg_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')

    def run_dataset(self):
        dataset = USDataset(
            target=self.args.target,
            normalize=False, aug_mode='none'
        )

        results = self.predict_images([i.image for i in dataset.items])

        oo = []
        for (item, result) in zip(dataset.items, results):
            result = result.tolist()
            oo.append({
                'path': item.path,
                'test': int(item.test),
                'gt': item.diag,
                'pred': float(result),
            })
        df = pd.DataFrame(oo)
        df.to_excel(f'out/{self.model_name}/report_{self.args.target}.xlsx', index=False)

    def load_images_from_dir_or_file(self, src):
        paths = []
        if os.path.isdir(src):
            paths = os.path.join(src, '*.jpg') + os.path.join(src, '*.png')
            images = [Image.open(p) for p in paths]
        elif os.path.isfile(src):
            paths = [src]
            images = [Image.open(src)]

        if len(images) == 0:
            raise RuntimeError(f'Invalid src: {src}')

        return images, paths

    def arg_predict(self, parser):
        parser.add_argument('--src', '-s', required=True)

    def run_predict(self):
        images, paths = self.load_images_from_dir_or_file(self.args.src)

        results = self.predict_images(images)

        for (path, result) in zip(paths, results):
            print(f'{path}: {result:.2f}')

if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
