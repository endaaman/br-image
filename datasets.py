import os
import os.path
import re
import shutil
from glob import glob
from typing import NamedTuple, Callable
from collections import OrderedDict
from endaaman import Commander, pad_to_size
from endaaman.torch import pil_to_tensor, tensor_to_pil

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


class BaseDataset(Dataset):
    def __init__(self, target='all', aug_mode='same', size=512,
                 normalize=True, train_test=0.25, len_scale=1, seed=42):
        self.target = target
        self.size = size
        self.train_test = train_test
        self.seed = seed

        # margin = size//20
        augs = {}
        augs['train'] = [
            # A.CenterCrop(width=size, height=size),
            # A.RandomCrop(width=size, height=size),
            # A.Resize(width=size, height=size),
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

            A.HueSaturationValue(p=0.3),
        ]

        augs['test'] = [
            A.Resize(width=size, height=size),
            # A.CenterCrop(width=size, height=size),
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
        if isinstance(self.train_test, str):
            df_all = df_all[df_all['id'] > 0]
            df_sp = pd.read_excel(self.train_test, index_col=0)
            df_all = pd.merge(df_all.reset_index(), df_sp, on='id')

            df_all = df_all.set_index('name')
            df_train = df_all[df_all['test'] < 1]
            df_test = df_all[df_all['test'] > 0]
        else:
            df_train, df_test = train_test_split(
                df_all,
                test_size=self.train_test,
                stratify=df_all.diagnosis,
                random_state=self.seed)

            df_test['test'] = 1
            df_train['test'] = 0
            df_all.loc[df_test.index, 'test'] = 1

        self.df = {
            'all': df_all,
            'test': df_test,
            'train': df_train
        }[self.target]

        self.items = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            item = self.load_item(idx, row)
            if item:
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        raise RuntimeError('Do override')


class SegDataset(BaseDataset):
    def load_item(self, idx, row):
        img = Image.open(f'data/crop/pe/{idx}_pe.png').copy()
        mask = Image.open(f'data/crop/m/{idx}_m.png').copy()
        return MaskItem(id=idx,
                        image=img,
                        mask=mask,
                        test=row['test'])

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        auged = self.albu(
            image=np.array(item.image),
            mask=np.array(item.mask),
        )
        x = auged['image']
        y = auged['mask'][None].float() / 255

        return x, y


class PEMDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.mode = kwargs.pop('mode') # ['p', 'e', 'pe', 'pem']
        super().__init__(**kwargs)

    def load_item(self, idx, row):
        img = Image.open(f'data/crop/{self.mode}/{idx}_{self.mode}.png').copy()
        return Item(id=idx,
                    image=img,
                    diagnosis=row['diagnosis'],
                    test=row['test'])

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.diagnosis])
        return x, y


class CroppedDataset(BaseDataset):
    def load_item(self, idx, row):
        mask = np.array(Image.open(f'data/crop/m/{idx}_m.png'))
        if not np.any(mask):
            # skip if mask is empty
            return None
        h = np.where(np.sum(mask, axis=1))[0][[0, -1]]
        w = np.where(np.sum(mask, axis=0))[0][[0, -1]]
        pe = Image.open(f'data/crop/pe/{idx}_pe.png')
        cropped = pe.crop((w[0], h[0], w[1], h[1]))
        # bg = Image.new('RGB', (self.size, self.size))
        # paste_center(bg, cropped)
        # if bg.width < cropped.width:
        #     print('w', idx, bg.width, cropped.width)
        # if bg.height < cropped.height:
        #     print('h', idx, bg.height, cropped.height)

        img = pad_to_size(cropped, size=self.size)
        return Item(id=idx,
                    image=img,
                    diagnosis=row['diagnosis'],
                    test=row['test'])

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.diagnosis])
        return x, y



class C(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--aug', '-a', default='same', choices=['same', 'train', 'test'])
        parser.add_argument('--mode', '-m', default='pem', choices=['pe', 'pem', 'pem_', 'seg', 'crop'])
        parser.add_argument('--size', '-s', type=int, default=512)
        # parser.add_argument('--a-flip', action='store_true')
        # parser.add_argument('--a-rotate', type=int, default=10)
        # parser.add_argument('--a-shrink', type=float, default=0.3)

    def pre_common(self):
        if self.a.mode == 'seg':
            self.ds = SegDataset(
                target=self.args.target,
                aug_mode=self.args.aug,
                size=self.args.size,
                normalize=self.args.function != 'samples',
            )
        elif self.a.mode == 'crop':
            self.ds = CroppedDataset(
                target=self.args.target,
                aug_mode=self.args.aug,
                size=self.args.size,
                normalize=self.args.function != 'samples',
            )
        else:
            self.ds = PEMDataset(
                mode=self.args.mode,
                target=self.args.target,
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
        dest = f'tmp/samples_{self.a.mode}/{d}'
        os.makedirs(dest, exist_ok=True)
        print(f'dest {dest}')
        total = self.args.length or len(self.ds)
        i = 0
        for x, y in tqdm(self.ds, total=total):
            if i >= total:
                break
            self.x = x
            self.y = y
            item = self.ds.items[i]
            img = tensor_to_pil(x)
            img.save(f'{dest}/{item.id}_{int(y)}.png')
            i += 1

    def arg_t(self, parser):
        parser.add_argument('--index', '-i', type=int, default=0)
        parser.add_argument('--id', '-d', type=str)

    def run_t(self):
        i = 0
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            self.item = self.ds.items[i]
            if self.a.id:
                if self.a.id == self.item.id:
                    break
            else:
                if self.a.index == i:
                    break
            i += 1

if __name__ == '__main__':
    c = C()
    c.run()
