import os
import os.path
import re
import shutil
from glob import glob
from typing import NamedTuple, Callable
from collections import OrderedDict
from endaaman import Commander
from endaaman.torch import calc_mean_and_std, pil_to_tensor, tensor_to_pil

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.crops.functional import center_crop


ImageFile.LOAD_TRUNCATED_IMAGES = True

class Item(NamedTuple):
    id: int
    diagnosis: bool
    image: ImageType

E_MEAN = 0.1820
E_STD  = 0.1372
P_MEAN = 0.1217
P_STD  = 0.1659
EP_MEAN = (E_MEAN + P_MEAN) / 2
EP_STD = (E_STD + P_STD) / 2


class MaximumSquareCrop(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        side = min(img.shape[:2])
        return center_crop(img, side, side)


SCALE = 1

class USDataset(Dataset):
    def __init__(self, target='all', size=256, normalize=True, seed=42, test_ratio=0.3):
        self.target = target
        self.size = size
        self.seed = seed
        self.test_ratio = test_ratio

        train_augs = [
            A.RandomResizedCrop(width=size, height=size, scale=[0.7, 1.0]),
            # A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),

            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            # A.HueSaturationValue(p=0.3),
        ]

        test_augs = [
            MaximumSquareCrop(),
            A.Resize(size, size),
        ]

        if normalize:
            common_augs = [
                A.Normalize(mean=[E_MEAN, P_MEAN, EP_MEAN], std=[E_STD, P_STD, EP_STD]),
                ToTensorV2(),
            ]
        else:
            common_augs = [ToTensorV2()]

        if target == 'test':
            self.albu = A.Compose(test_augs + common_augs)
        else:
            self.albu = A.Compose(train_augs + common_augs)

        self.load_data()

    def load_data(self):
        df_all = pd.read_excel('data/master.xlsx', index_col=0)

        if self.target == 'all':
            df = df_all
        else:
            df_train, df_test = train_test_split(
                df_all,
                test_size=self.test_ratio,
                stratify=df_all.diagnosis,
                random_state=self.seed)

            if self.target == 'test':
                df = df_test
            elif self.target == 'train':
                df = df_train
            else:
                raise ValueError('Invalid target', self.target)

        self.df = df
        self.items = []
        for idx, row in self.df.iterrows():
            img = Image.open(f'data/pe/{idx}_pe.png')
            self.items.append(Item(id=idx,
                                   diagnosis=row['diagnosis'],
                                   image=img))

    def __len__(self):
        return len(self.items) * SCALE

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        t = self.albu(image=np.array(item.image))['image']
        return t, torch.FloatTensor([item.diagnosis])


class C(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--size', '-s', type=int, default=256)
        parser.add_argument('--a-flip', action='store_true')
        parser.add_argument('--a-rotate', type=int, default=10)
        parser.add_argument('--a-shrink', type=float, default=0.3)

    def pre_common(self):
        self.ds = USDataset(
            target=self.args.target,
            size=self.args.size,
            normalize=self.args.function != 'samples',
        )

    def run_samples(self):
        t = self.args.target
        d = f'tmp/samples_{t}'
        os.makedirs(d, exist_ok=True)
        for i, (x, y) in tqdm(enumerate(self.ds), total=len(self.ds)):
            if i > len(self.ds):
                break
            self.x = x
            self.y = y
            img = tensor_to_pil(x)
            img.save(f'{d}/{i}_{int(y)}.png')

    def run_t(self):
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            print(y, x.shape)
            self.i = tensor_to_pil(x)
            break

if __name__ == '__main__':
    # ds = USDataset()
    # for (x, y) in ds:
    #     print(x.shape)
    #     print(y)
    #     break
    c = C()
    c.run()
