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

    def arg_split(self, parser):
        parser.add_argument('--ratio', '-r', type=float, default=0.3)

    def run_split(self):
        # cache label data
        df = pd.read_excel('data/master.xlsx', index_col=0).dropna()
        df_train, df_test = train_test_split(df, test_size=self.args.ratio, stratify=df.diagnosis)
        df['test'] = 0
        df.at[df_test.index, 'test'] = 1
        os.makedirs('data/cache', exist_ok=True)
        p = 'data/cache/labels.csv'
        df.to_csv(p)

        # cache images
        for f in glob('data/images/*.png'):
            basename = os.path.basename(f)

        print(f'wrote {p}')
