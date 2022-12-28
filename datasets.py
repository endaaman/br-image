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
from albumentations.augmentations.crops.functional import center_crop, random_crop

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Item(NamedTuple):
    id: int
    image: ImageType
    diagnosis: bool
    test: bool

class MaskItem(NamedTuple):
    id: int
    image: ImageType
    mask: ImageType
    test: bool

E_MEAN = 0.1820
E_STD  = 0.1372
P_MEAN = 0.1217
P_STD  = 0.1659
EP_MEAN = (E_MEAN + P_MEAN) / 2
EP_STD = (E_STD + P_STD) / 2

MEAN = [E_MEAN, P_MEAN, EP_MEAN]
STD = [E_STD, P_STD, EP_STD]


class MaximumSquareCenterCrop(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        side = min(img.shape[:2])
        return center_crop(img, side, side)

    def get_transform_init_args_names(self):
        return ()

class MaximumSquareRandomCrop(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        side = min(img.shape[:2])
        return random_crop(img, side, side, 0, 0)

    def get_transform_init_args_names(self):
        return ()


class USDataset(Dataset):
    def __init__(self, target='all', mode='pem', aug_mode='same', size=512,
                 normalize=True, test_ratio=0.25, len_scale=1, seed=42):
        self.target = target
        self.mode = mode # ['p', 'e', 'pe', 'pem', 'seg']
        self.size = size
        self.test_ratio = test_ratio
        self.len_scale = len_scale
        self.seed = seed

        # margin = size//20
        augs = {}
        augs['train'] = [
            A.Resize(width=size, height=size),
            A.RandomResizedCrop(width=size, height=size, scale=[0.9, 1.1]),

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

        augs['test'] = [
            A.Resize(width=size, height=size),
        ]

        augs['all'] = augs['test']

        # select aug
        if aug_mode == 'same':
            aug = augs[target]
        elif aug_mode == 'none':
            aug = []
        else:
            aug = augs[aug_mode]

        if normalize:
            aug += [A.Normalize(mean=MEAN, std=STD)]
        aug += [ToTensorV2()]

        self.albu = A.Compose(aug)
        self.load_data()

    def load_data(self):
        df_all = pd.read_excel('data/label.xlsx', index_col=0)
        df_all['test'] = 0
        df_train, df_test = train_test_split(
            df_all,
            test_size=self.test_ratio,
            stratify=df_all.diagnosis,
            random_state=self.seed)

        df_test['test'] = 1
        # df_train['test'] = 0
        df_all.loc[df_test.index, 'test'] = 1

        if self.target == 'all':
            df = df_all
        elif self.target == 'test':
            df = df_test
        elif self.target == 'train':
            df = df_train
        else:
            raise ValueError('Invalid target', self.target)

        self.df = df
        self.items = []

        if self.mode == 'seg':
            for idx, row in self.df.iterrows():
                img = Image.open(f'data/crop/pe/{idx}_pe.png').copy()
                mask = Image.open(f'data/crop/m/{idx}_m.png').copy()
                self.items.append(
                    MaskItem(id=idx,
                         image=img,
                         mask=mask,
                         test=row['test']))
        else:
            for idx, row in self.df.iterrows():
                img = Image.open(f'data/crop/{self.mode}/{idx}_{self.mode}.png').copy()
                self.items.append(
                    Item(id=idx,
                         image=img,
                         diagnosis=row['diagnosis'],
                         test=row['test']))

    def __len__(self):
        return int(len(self.items) * self.len_scale)

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        if self.mode == 'seg':
            auged = self.albu(
                image=np.array(item.image),
                mask=np.array(item.mask),
            )
            x = auged['image']
            y = auged['mask'][None].float() / 255
        else:
            x = self.albu(image=np.array(item.image))['image']
            y = torch.FloatTensor([item.diagnosis])

        return x, y



class C(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--aug', '-a', default='same', choices=['same', 'train', 'test'])
        parser.add_argument('--mode', '-m', default='pem', choices=['pe', 'pem', 'pem_', 'seg'])
        parser.add_argument('--size', '-s', type=int, default=256)
        # parser.add_argument('--a-flip', action='store_true')
        # parser.add_argument('--a-rotate', type=int, default=10)
        # parser.add_argument('--a-shrink', type=float, default=0.3)

    def pre_common(self):
        self.ds = USDataset(
            target=self.args.target,
            mode=self.args.mode,
            aug_mode=self.args.aug,
            size=self.args.size,
            normalize=self.args.function != 'samples',
        )

    def arg_samples(self, parser):
        parser.add_argument('--length', '-l', type=int)
        parser.add_argument('--dest', '-d')

    def run_samples(self):
        t = self.args.target
        d = self.args.dest or t
        dest = f'out/samples/{d}'
        os.makedirs(dest, exist_ok=True)
        total = self.args.length or len(self.ds)
        for i, (x, y) in tqdm(enumerate(self.ds), total=total):
            if i > total:
                break
            self.x = x
            self.y = y
            img = tensor_to_pil(x)
            img.save(f'{dest}/{i}_{int(y)}.png')

    def run_t(self):
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            tensor_to_pil(x).save('tmp/x.png')
            tensor_to_pil(y).save('tmp/y.png')
            break

if __name__ == '__main__':
    c = C()
    c.run()
