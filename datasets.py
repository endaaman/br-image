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


class Item(NamedTuple):
    id: int
    diagnosis: bool
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


class USDataset(Dataset):
    def __init__(self, test=False, size=256, a_flip=False, a_rotate=0, a_shrink=0):
        self.test = test
        self.size = size
        self.a_flip = a_flip
        self.a_rotate = a_rotate
        self.a_shrink = a_shrink
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

            enhance_image = original_image.crop(rule.enhance_roi.rect()).convert('L')
            plain_image = original_image.crop(rule.plain_roi.rect()).convert('L')
            assert enhance_image.size == plain_image.size
            self.items.append(Item(idx, row['diagnosis'],
                                   original_image, enhance_image, plain_image))

    def __len__(self):
        return len(self.items)

    def aug(self, item, a_flip, a_shrink, a_rotate):
        angle = np.random.randint(0, a_rotate*2+1) - a_rotate
        enhance_image = item.enhance_image.rotate(angle, expand=False)
        plain_image = item.plain_image.rotate(angle, expand=False)

        mini_side = min(plain_image.size)
        shrink = np.random.rand() * a_shrink
        offset = round(mini_side * shrink)
        size = mini_side - offset
        x = np.random.randint(0, max(plain_image.width - size, 1))
        y = np.random.randint(0, max(plain_image.height - size, 1))
        rect = (x, y, x + size, y + size)
        enhance_image = enhance_image.crop(rect)
        plain_image = plain_image.crop(rect)

        e = np.array(enhance_image.resize((self.size, self.size)))
        p = np.array(plain_image.resize((self.size, self.size)))
        dummy = np.zeros((self.size, self.size), dtype=np.uint8)
        ep = np.stack([e, p, dummy], 0)

        # normalize
        ep[:, :, 0] = (ep[:, :, 0] - E_MEAN) / E_STD
        ep[:, :, 1] = (ep[:, :, 1] - P_MEAN) / P_STD
        t = torch.from_numpy(ep) / 255

        if a_flip and 0.5 < np.random.rand():
            t = torch.flip(t, (2, ))

        return t

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.test:
            t = self.aug(item, False, 0, 0)
        else:
            t = self.aug(item, self.a_flip, self.a_shrink, self.a_rotate)

        return t, torch.FloatTensor([item.diagnosis])



class C(Commander):
    def arg_split(self, parser):
        parser.add_argument('--ratio', '-r', type=float, default=0.3)

    def run_split(self):
        df = pd.read_excel('data/master.xlsx', index_col=0).dropna()
        df_train, df_test = train_test_split(df, test_size=self.args.ratio, stratify=df.diagnosis)
        df['test'] = 0
        df.at[df_test.index, 'test'] = 1
        p = 'data/cache/labels.tsv'
        df.to_csv(p, sep='\t')
        print(f'wrote {p}')

    def arg_common(self, parser):
        parser.add_argument('--test', '-t', action='store_true')
        parser.add_argument('--flip', action='store_true')
        parser.add_argument('--rotate', type=int, default=10)
        parser.add_argument('--shrink', type=float, default=0.3)

    def pre_common(self):
        self.ds = USDataset(
            test=self.args.test,
            a_flip=self.args.flip,
            a_shrink=self.args.shrink,
            a_rotate=self.args.rotate,
        )

    def run_dump_ep(self):
        ds = USDataset()
        for i in tqdm(ds.items):
            i.enhance_image.save(f'out/ep/{i.id:03}e.png')
            i.plain_image.save(f'out/ep/{i.id:03}p.png')

            e = np.array(i.enhance_image)
            p = np.array(i.plain_image)
            dummy = np.zeros(e.size, dtype=np.uint8).reshape(e.shape)
            pe = Image.fromarray(np.stack([e, p, dummy], 2))
            pe.save(f'out/ep/{i.id:03}ep.png')

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
