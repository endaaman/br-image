import os
import os.path
import re
import shutil
from glob import glob
from typing import NamedTuple, Callable
from endaaman import Commander
from endaaman.torch import calc_mean_and_std, pil_to_tensor, tensor_to_pil

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from PIL.Image import Image as img
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.crops.functional import center_crop


class Item(NamedTuple):
    id: int
    diagnosis: bool
    image: img
    original_image: img
    plain_image: img
    enhance_image: img

class ROI(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

class ROIRule(NamedTuple):
    code: str
    enhance_roi: ROI
    plain_roi: ROI

roi_rules = (
    # VERTICAL
    ROIRule(
        'ver_C', # (1280, 960)
        ROI(345, 169, 591, 293), # [1] h:169
        ROI(345, 588, 591, 293),
    ),

    # HORIZONTAL
    ROIRule(
        'hor_A', # (960, 720)
        ROI(62, 87, 417, 495),
        ROI(479, 87, 417, 495),
    ),
    ROIRule(
        'hor_B', # (1280, 720)
        ROI(18, 102, 554, 435),
        ROI(605, 102, 554, 435),
    ),

    ROIRule(
        'hor_C', # (1280, 960)
        ROI(116, 144, 520, 658),
        ROI(640, 144, 520, 658),
    ),

    ROIRule(
        'hor_X', # (1552, 873)
        ROI(74, 109, 650, 570),
        ROI(724, 109, 650, 570),
    ),
)

E_MEAN = 0.1820
E_STD  = 0.1372
P_MEAN = 0.1217
P_STD  = 0.1659



class MaximumSquareCrop(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        side = min(img.shape[:2])
        return center_crop(img, side, side)


SCALE = 1

class USDataset(Dataset):
    def __init__(self, test=False, size=256, normalize=True):
        print('normalize', normalize)
        self.test = test
        self.size = size

        train_augs = [
            A.RandomResizedCrop(width=size, height=size, scale=[0.7, 1.0]),
            A.Resize(size, size),
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
                A.Normalize(mean=[E_MEAN, P_STD, 1], std=[E_MEAN, P_STD, 1]),
                ToTensorV2(),
            ]
        else:
            common_augs = [ToTensorV2()]

        if test:
            self.albu = A.Compose(test_augs + common_augs)
        else:
            self.albu = A.Compose(train_augs + common_augs)

        self.load_data()

    def load_data(self):
        df = pd.read_csv('data/cache/labels.tsv', index_col=0, sep='\t')
        df = df[df['test'] == self.test]

        self.df = df
        self.items = []
        for idx, row in self.df.iterrows():
            p = f'data/images/{idx:03}.png'
            original_image = Image.open(p)
            rule = None
            for r in roi_rules:
                if r.code == row['image_type']:
                    rule = r
            if not rule:
                raise ValueError(f'{p} is unknown image type.')

            plain_image = original_image.crop(rule.enhance_roi.rect()).convert('L')
            enhance_image = original_image.crop(rule.plain_roi.rect()).convert('L')
            assert enhance_image.size == plain_image.size
            base_size = plain_image.size
            if row['swap']:
                enhance_image, plain_image = plain_image, enhance_image
            e = np.array(enhance_image.resize(base_size))
            p = np.array(plain_image.resize(base_size))
            # reverse dimension
            dummy = np.zeros(base_size[::-1], dtype=np.uint8)
            ep = np.stack([e, p, dummy], 0).transpose((1, 2, 0))
            ep = Image.fromarray(ep)

            self.items.append(Item(idx, row['diagnosis'],
                                   ep, original_image, enhance_image, plain_image))

    def __len__(self):
        return len(self.items) * SCALE

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        t = self.albu(image=np.array(item.image))['image']
        return t, torch.FloatTensor([item.diagnosis])


class C(Commander):
    def arg_split(self, parser):
        parser.add_argument('--ratio', '-r', type=float, default=0.3)

    def run_split(self):
        # cache label data
        df = pd.read_excel('data/master.xlsx', index_col=0).dropna()
        df_train, df_test = train_test_split(df, test_size=self.args.ratio, stratify=df.diagnosis)
        df['test'] = 0
        df.at[df_test.index, 'test'] = 1
        os.makedirs('data/cache', exist_ok=True)
        p = 'data/cache/labels.tsv'
        df.to_csv(p, sep='\t')

        # cache images
        for f in glob('data/images/*.png'):
            basename = os.path.basename(f)

        print(f'wrote {p}')

    def arg_common(self, parser):
        parser.add_argument('--test', '-t', action='store_true')
        parser.add_argument('--flip', action='store_true')
        parser.add_argument('--rotate', type=int, default=10)
        parser.add_argument('--shrink', type=float, default=0.3)

    def pre_common(self):
        self.ds = USDataset(
            test=self.args.test,
            normalize=self.args.function != 'samples',
        )

    def run_dump_ep(self):
        for i in tqdm(self.ds.items):
            i.plain_image.save(f'out/ep/{i.id:03}_0p.png')
            i.enhance_image.save(f'out/ep/{i.id:03}_1e.png')
            i.image.save(f'out/ep/{i.id:03}_2ep.png')

    def run_mean_std(self):
        e_mean, e_std = calc_mean_and_std([item.enhance_image for item in self.ds.items])
        print('e_mean', e_mean)
        print('e_std', e_std)

        p_mean, p_std = calc_mean_and_std([item.plain_image for item in self.ds.items])
        print('p_mean', p_mean)
        print('p_std', p_std)

    def run_samples(self):
        t = 'test' if self.args.test else 'train'
        d = f'tmp/samples_{t}'
        os.makedirs(d, exist_ok=True)
        for i, (x, y) in enumerate(self.ds):
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
    c = C(options={'no_pre_common': ['split']})
    c.run()
