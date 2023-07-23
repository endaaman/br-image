import os
import os.path
import re
import shutil
from glob import glob
from typing import NamedTuple, Callable
from collections import OrderedDict

from pydantic import Field
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

from endaaman import pad_to_size
from endaaman.ml import pil_to_tensor, tensor_to_pil, get_global_seed, BaseMLCLI

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Item(NamedTuple):
    name: str
    id: int
    image: ImageType
    diagnosis: bool
    test: bool

class MaskItem(NamedTuple):
    name: str
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
                 normalize=True, split=0.25, seed=get_global_seed()):
        self.target = target
        self.size = size
        self.split = split
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
        df_all = df_all[df_all['skip'] < 1]
        if isinstance(self.split, str):
            df_sp = pd.read_excel(self.split, index_col=0)
            # left join
            df_all = df_all \
                .reset_index() \
                .merge(df_sp, left_on='id_head', right_on='id', how='left') \
                .set_index('name')
            # if NA, set train
            df_all.loc[df_all['test'].isna(), 'test'] = 0
            df_train = df_all[df_all['test'] < 1]
            df_test = df_all[df_all['test'] > 0]
        else:
            df_train, df_test = train_test_split(
                df_all,
                test_size=self.split,
                stratify=df_all['diagnosis'],
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

        for name, row in tqdm(self.df.iterrows(), total=len(self.df)):
            item = self.load_item(name, row)
            if item:
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        raise RuntimeError('Do override')


class SegDataset(BaseDataset):
    def load_item(self, name, row):
        img = Image.open(f'data/crop/pe/{name}_pe.png').copy()
        mask = Image.open(f'data/crop/m/{name}_m.png').copy()
        return MaskItem(name=name,
                        id=row.id,
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

    def load_item(self, name, row):
        img = Image.open(f'data/crop/{self.mode}/{name}_{self.mode}.png').copy()
        return Item(name=name,
                    id=row.id,
                    image=img,
                    diagnosis=row['diagnosis'],
                    test=row['test'])

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.diagnosis])
        return x, y


class CroppedDataset(BaseDataset):
    def load_item(self, name, row):
        mask = np.array(Image.open(f'data/crop/m/{name}_m.png'))
        if not np.any(mask):
            # skip if mask is empty
            return None
        h = np.where(np.sum(mask, axis=1))[0][[0, -1]]
        w = np.where(np.sum(mask, axis=0))[0][[0, -1]]
        pe = Image.open(f'data/crop/pe/{name}_pe.png')
        cropped = pe.crop((w[0], h[0], w[1], h[1]))
        # bg = Image.new('RGB', (self.size, self.size))
        # paste_center(bg, cropped)
        # if bg.width < cropped.width:
        #     print('w', name, bg.width, cropped.width)
        # if bg.height < cropped.height:
        #     print('h', name, bg.height, cropped.height)

        img = pad_to_size(cropped, size=self.size)
        return Item(name=name,
                    id=row.id,
                    image=img,
                    diagnosis=row['diagnosis'],
                    test=row['test'])

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.diagnosis])
        return x, y



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        target:str = Field('all', cli=('--target', '-t'), regex=r'^all|train|test$')
        aug:str = Field('same', cli=('--aug', '-a'), regex=r'^same|train|test|none$')
        mode:str = Field('pem', cli=('--mode', '-m'), regex=r'^pe|pem|pem_|seg|crop$')
        size:int = Field(512, cli=('--size', '-s'))

    def pre_common(self, a):
        if a.mode == 'seg':
            self.ds = SegDataset(
                target=a.target,
                aug_mode=a.aug,
                size=a.size,
                normalize=a.function != 'samples',
            )
        elif a.mode == 'crop':
            self.ds = CroppedDataset(
                target=a.target,
                aug_mode=a.aug,
                size=a.size,
                normalize=a.function != 'samples',
            )
        else:
            self.ds = PEMDataset(
                mode=a.mode,
                target=a.target,
                aug_mode=a.aug,
                size=a.size,
                normalize=a.function != 'samples',
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
            img.save(f'{dest}/{item.name}_{int(y)}.png')
            i += 1

    class TArgs(CommonArgs):
        index:int = Field(0, cli=('-i', ))
        id:str = ''

    def run_t(self):
        i = 0
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            self.item = self.ds.items[i]
            if self.a.id:
                if self.a.id == self.item.name:
                    break
            else:
                if self.a.index == i:
                    break
            i += 1

if __name__ == '__main__':
    # cli = CLI()
    # cli.run()

    ds = PEMDataset(
        mode='pem',
        target='all',
        aug_mode='none',
        size=512,
        normalize=False,
    )
