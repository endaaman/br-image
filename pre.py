import os
import os.path
import re
import shutil
from glob import glob

from tqdm import tqdm
import pandas as pd
from endaaman.torch import Trainer

from typing import NamedTuple, Callable
from PIL import Image
from PIL.Image import Image as img
from torch.utils.data import Dataset

class C(Trainer):
    def run_pre(self):
        ll = sorted(glob('data/contrastUS/*/*.png'))
        for s in tqdm(ll):
            i = os.path.dirname(s).split('/')[-1]
            shutil.copyfile(s, f'data/images/{i}.png')

    def run_hi(self):
        ds = EchoDataset()
        for i in ds.items:
            t = None
            if i.original_image.size == (960, 720):
                t = 'hor_A'
            if i.original_image.size == (1280, 720):
                t = 'hor_B'
            if i.original_image.size == (1280, 960):
                t = 'ver_C'
            if i.original_image.size == (1552, 873):
                t = 'hor_X'
            ds.df.at[i.id, 'image_type'] = t
        ds.df.to_csv('l.csv')
