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
    def __init__(self, test=False, size=256):
        self.size = size
        self.load_data()

    def load_data(self, ):
        self.df = pd.read_excel('data/labels.xlsx', index_col=0)

        self.items = []
        paths = sorted(glob(f'data/images/*.png'))
        for p in paths:
            basename = os.path.basename(p)
            m = re.match(r'^([0-9]{3})\.png$', basename)
            if not m:
                raise ValueError(f'{p} is invalid formatting.')
            idx = int(m[1]) # trim zeros
            row = self.df.loc[idx]
            original_image = Image.open(p)
            rule = None
            for r in roi_rules:
                if r.code == row['image_type']:
                    rule = r
                    break
            if not rule:
                raise ValueError(f'{p} is unknown image type.')

            enhance_image = original_image.crop(rule.enhance_roi.rect()).convert('L')
            plain_image = original_image.crop(rule.plain_roi.rect()).convert('L')
            assert enhance_image.size == plain_image.size
            self.items.append(Item(idx, row['diagnosis'],
                                   original_image, enhance_image, plain_image))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        MAX_SHRINK = 0.3

        item = self.items[idx]

        mini_side = min(item.plain_image.size)
        shrink = np.random.rand() * MAX_SHRINK
        offset = round(mini_side * shrink)
        size = mini_side - offset
        x = np.random.randint(0, item.plain_image.width - size)
        y = np.random.randint(0, item.plain_image.height - size)
        rect = (x, y, x + size, y + size)
        enhance_image = item.enhance_image.crop(rect)
        plain_image = item.plain_image.crop(rect)

        e = np.array(enhance_image.resize((self.size, self.size)))
        p = np.array(plain_image.resize((self.size, self.size)))
        dummy = np.zeros((self.size, self.size), dtype=np.uint8)
        ep = np.stack([e, p, dummy], 0)

        # normalize
        ep[:, :, 0] = (ep[:, :, 0] - E_MEAN) / E_STD
        ep[:, :, 1] = (ep[:, :, 1] - P_MEAN) / P_STD
        t = torch.from_numpy(ep) / 255

        if 0.5 < np.random.rand():
            t = torch.flip(t, (2, ))

        return t, item.diagnosis



class C(Commander):
    def pre_common(self):
        self.ds = USDataset()

    def run_dump_ep(self):
        for i in tqdm(self.ds.items):
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

    def run_t(self):
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            print(y, x.shape)
            break

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
