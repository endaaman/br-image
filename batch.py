import os
import os.path
import re
import shutil
from glob import glob
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import NamedTuple, Callable
from PIL import Image
from PIL.Image import Image as img
from torch.utils.data import Dataset
import pydicom

from endaaman import Commander, with_wrote, with_log



class ROI(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

class ROIRule(NamedTuple):
    enhance_roi: ROI
    plain_roi: ROI

roi_rules = {
    # VERTICAL
    'C_ver': ROIRule( # (1280, 960)
        ROI(345, 169, 591, 293), # [1] h:169
        ROI(345, 588, 591, 293),
    ),

    # HORIZONTAL
    'A_hor': ROIRule( # (960, 720)
        ROI(62, 87, 417, 495),
        ROI(479, 87, 417, 495),
    ),

    'B_hor': ROIRule( # (1280, 720)
        ROI(18, 102, 554, 435),
        ROI(605, 102, 554, 435),
    ),

    'C_hor': ROIRule( # (1280, 960)
        ROI(116, 144, 520, 658),
        ROI(640, 144, 520, 658),
    ),

    'C_hor2': ROIRule( # (1280, 960)
        ROI(40, 141, 560, 680),
        ROI(680, 141, 560, 680),
    ),

    'D_hor': ROIRule( # (1552, 873)
        ROI(74, 109, 650, 570),
        ROI(724, 109, 650, 570),
    ),

    'D_hor2': ROIRule( # (1552, 873)
        ROI(2, 133, 720, 636),
        ROI(724, 133, 720, 636),
    ),

    'D_hor3': ROIRule( # (1552, 873)
        ROI(45, 129, 640, 640),
        ROI(766, 129, 640, 640),
    ),
}

def get_default_style_by_image(img):
    if img.size == (960, 720):
        t = 'A_hor'
    elif img.size == (1280, 720):
        t = 'B_hor'
    elif img.size == (1280, 960):
        t = 'C_ver'
    elif img.size == (1552, 873):
        t = 'D_hor'
    else:
        t = 'unknown'
    return t


class C(Commander):
    def arg_convert_dicom(self, parser):
        parser.add_argument('--src', default='data/dicom')
        parser.add_argument('--dest', default='out/from_dicom')

    def run_convert_dicom(self):
        os.makedirs(self.args.dest, exist_ok=True)
        pattern = os.path.join(self.args.src, '*/*')
        paths = glob(pattern, recursive=True)
        data = defaultdict(list)

        for p in paths:
            # path is like
            # data/dicom/001/jMAC.1.2.392.<SNIP>5.2.602
            m = re.match(f'^{self.args.src}' + r'/((ba|a)?[0-9]{3}).*/.*$', p)
            if not m:
                print('not match: ', p)
                continue

            data[m[1]].append(p)

        for (id, paths) in tqdm(data.items()):
            for i, p in enumerate(paths):
                dcm = pydicom.read_file(p)
                # BGR -> RGB
                img = Image.fromarray(dcm.pixel_array[:, :, [2, 1, 0]])
                img.save(os.path.join(self.args.dest, f'{id}_{i}.png'))

    def arg_detect_style(self, parser):
        parser.add_argument('--src', default='data/images')

    def run_detect_style(self):
        paths = sorted(glob(os.path.join(self.args.src, '*.png')))

        data = []
        for p in paths:
            m = re.match(r'.*/((ba|a)?[0-9]{3})_0\.png', p)
            if not m:
                raise ValueError('Invalid filename: ', p)
            id = m[1]
            print(id, p)
            img = Image.open(p)
            t = get_default_style_by_image(img)
            data.append({
                'id': id,
                'style': t,
            })

        pd.DataFrame(data).to_csv(with_wrote('out/types.csv'), index=False)


    def run_check_master(self):
        df = pd.read_excel('data/master.xlsx', index_col=0)

        paths = sorted(glob('data/images/*.png'))

        data = []
        print('checking images')
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            if not name in df.index:
                print(f'{p} exists but {name} is not registered to df')

        print('checking df')
        for idx, row in df.iterrows():
            path = f'data/images/{idx}.png'
            if not os.path.exists(path):
                print(f'{name} is not registered but {p} does not exist')

        print('done')

    def arg_cache(self, parser):
        parser.add_argument('--src', default='data/images/')
        parser.add_argument('--dest', default='data/cache/')
        parser.add_argument('--target', '-t', type=str, nargs='+', default=[])
        parser.add_argument('--swap', action='store_true')

    def run_cache(self):
        df = pd.read_excel('data/master.xlsx', index_col=0).dropna()
        os.makedirs(self.args.dest, exist_ok=True)

        if len(self.args.target) > 0:
            df = df.loc[self.args.target]

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            rule = roi_rules.get(f'{row.resolution}_{row.position}', None)
            if not rule:
                raise ValueError(f'{p} is unknown image type.')

            original_image = Image.open(f'data/images/{idx}.png')

            plain_image = original_image.crop(rule.plain_roi.rect()).convert('L')
            enhance_image = original_image.crop(rule.enhance_roi.rect()).convert('L')
            assert enhance_image.size == plain_image.size
            base_size = plain_image.size
            if row['swap'] ^ self.args.swap:
                enhance_image, plain_image = plain_image, enhance_image
            e = np.array(enhance_image)
            p = np.array(plain_image)
            # reverse dimension
            dummy = np.zeros(base_size[::-1], dtype=np.uint8)
            pe = np.stack([p, e, dummy], 0).transpose((1, 2, 0))
            pe = Image.fromarray(pe)

            dest = lambda code: os.path.join(self.args.dest, f'{idx}_{code}.png')
            # plain_image.save(dest('p'))
            # enhance_image.save(dest('e'))
            pe.save(dest('pe'))


c = C()
c.run()
