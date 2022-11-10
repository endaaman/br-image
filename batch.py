import os
import os.path
import re
import shutil
from glob import glob
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
from typing import NamedTuple, Callable
from PIL import Image
from PIL.Image import Image as img
from torch.utils.data import Dataset
import pydicom

from endaaman.torch import Trainer

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

    def run_convert_dicom(self):
        base_dir = 'tmp/dicom'
        dest_dir = 'out/from_dicom'

        pattern = os.path.join(base_dir, '*/*')
        paths = glob(pattern, recursive=True)
        data = defaultdict(list)

        for p in paths:
            # path is like
            # data/dicom/001/jMAC.1.2.392.<SNIP>5.2.602
            m = re.match(f'^{base_dir}' + r'/([0-9]{3}).*/.*$', p)
            if not m:
                print('not match: ', p)
                continue

            data[m[1]].append(p)

        for (id, paths) in tqdm(data.items()):
            for i, p in enumerate(paths):
                dcm = pydicom.read_file(p)
                # BGR -> RGB
                img = Image.fromarray(dcm.pixel_array[:, :, [2, 1, 0]])
                img.save(os.path.join(dest_dir, f'{id}_{i}.png'))

c = C()
c.run()
