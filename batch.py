import os
import os.path
import re
import shutil
from glob import glob
from collections import defaultdict
from typing import NamedTuple, Callable

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset
import pydicom
from gimpformats.gimpXcfDocument import GimpDocument

from endaaman import Commander, with_wrote, with_log


class ROI(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

class ROIRule(NamedTuple):
    e_xy: tuple[int, int]
    p_xy: tuple[int, int]
    size: tuple[int, int]

    def e_rect(self):
        return (self.e_xy[0], self.e_xy[1], self.e_xy[0]+self.size[0], self.e_xy[1]+self.size[1])

    def p_rect(self):
        return (self.p_xy[0], self.p_xy[1], self.p_xy[0]+self.size[0], self.p_xy[1]+self.size[1])

roi_rules = {
    # VERTICAL
    'C_ver': ROIRule( # (1280, 960)
        # (256, 140),
        # (256, 560),
        # (796, 344),
        (345, 142),
        (345, 560),
        (590, 347)
    ),

    # HORIZONTAL
    'A_hor': ROIRule( # (960, 720)
        (62,  87),
        (479, 87),
        (417, 495),
    ),

    'B_hor': ROIRule( # (1280, 720)
        (18,  102),
        (605, 102),
        (554, 435)
    ),

    'C_hor': ROIRule( # (1280, 960)
        (116, 144),
        (640, 144),
        (520, 658)
    ),

    'C_hor2': ROIRule( # (1280, 960)
        (40, 141),
        (680, 141),
        (560, 680),
    ),

    'D_hor': ROIRule( # (1552, 873)
        (74, 109),
        (724, 109),
        (650, 570),
    ),

    'D_hor2': ROIRule( # (1552, 873)
        (2, 133),
        (724, 133),
        (720, 636),
    ),

    'D_hor3': ROIRule( # (1552, 873)
        (45, 129),
        (766, 129),
        (640, 640),
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


def pad2square(img, new_size=None):
    M = max(img.size)
    bg = Image.new(mode=img.mode, size=(M, M))
    x = (M - img.width) // 2
    y = (M - img.height) // 2
    bg.paste(img, (x, y))
    if new_size:
        bg = bg.resize((new_size, new_size))
    return bg

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

        for (id_, paths) in tqdm(data.items()):
            for i, p in enumerate(paths):
                dcm = pydicom.read_file(p)
                # BGR -> RGB
                img = Image.fromarray(dcm.pixel_array[:, :, [2, 1, 0]])
                img.save(os.path.join(self.args.dest, f'{id_}_{i}.png'))

    def arg_detect_style(self, parser):
        parser.add_argument('--src', default='data/images')

    def run_detect_style(self):
        paths = sorted(glob(os.path.join(self.args.src, '*.png')))

        data = []
        for p in paths:
            m = re.match(r'.*/((ba|a)?[0-9]{3})_0\.png', p)
            if not m:
                raise ValueError('Invalid filename: ', p)
            id_ = m[1]
            print(id_, p)
            img = Image.open(p)
            t = get_default_style_by_image(img)
            data.append({
                'id': id_,
                'style': t,
            })

        pd.DataFrame(data).to_csv(with_wrote('out/types.csv'), index=False)


    def run_check_label(self):
        df = pd.read_excel('data/label.xlsx', index_col=0)

        paths = sorted(glob('data/images/*.png'))

        print('checking images')
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            if not name in df.index:
                print(f'{p} exists but {name} is not registered to df')

        print('checking df')
        for idx, __row in df.iterrows():
            path = f'data/images/{idx}.png'
            if not os.path.exists(path):
                print(f'{name} is not registered but {path} does not exist')

        print('done')

    def arg_cache(self, parser):
        parser.add_argument('--src', default='data/images/')
        parser.add_argument('--dest', default='data/cache/')
        parser.add_argument('--target', '-t', type=str, nargs='+', default=[])
        parser.add_argument('--swap', action='store_true')
        parser.add_argument('--square', action='store_true')
        parser.add_argument('--from', type=int)
        parser.add_argument('--to', type=int)

    def run_cache(self):
        df = pd.read_excel('data/label.xlsx', index_col=0).dropna()
        os.makedirs(self.args.dest, exist_ok=True)

        if len(self.args.target) > 0:
            df = df.loc[self.args.target]

        total = len(df)
        for i, (idx, row) in (t:=tqdm(enumerate(df.iterrows()), total=total)):
            ii = int(re.match(r'.*(\d\d\d)_\d$', idx)[1])
            _from = getattr(self.args, 'from')
            if _from and ii < _from:
                continue
            _to = getattr(self.args, 'to')
            if _to and ii > _to:
                continue
            rule_code = f'{row.resolution}_{row.position}'
            rule = roi_rules.get(rule_code, None)
            if not rule:
                raise ValueError(f'{idx}({rule_code}) is unknown image type.')

            original_image = Image.open(f'data/images/{idx}.png')

            plain_image = original_image.crop(rule.p_rect()).convert('L')
            enhance_image = original_image.crop(rule.e_rect()).convert('L')
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

            if self.args.square:
                # e, p, pe = [pad2square(i, 720) for i in (e, p, ep) ]
                pe = pad2square(pe, 720)

            dest_fn = lambda code: os.path.join(self.args.dest, f'{idx}_{code}.png')
            # plain_image.save(dest('p'))
            # enhance_image.save(dest('e'))

            d = dest_fn('pe')
            pe.save(d)

            t.set_description(f'{d} {i}/{total}')
            t.refresh()


    def arg_extract_mask(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--dest', '-d', default='mask')
        parser.add_argument('--from', type=int)
        parser.add_argument('--to', type=int)

    def run_extract_mask(self):
        os.makedirs(self.args.dest, exist_ok=True)
        for i, p in tqdm(enumerate(glob(os.path.join(self.args.src, '*.xcf')))):
            _from = getattr(self.args, 'from')
            if _from and i < _from:
                continue
            _to = getattr(self.args, 'to')
            if _to and i < _to:
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            prj = GimpDocument(p)
            d = os.path.join(self.args.dest, f'{name}.png')
            # print(d)
            prj.layers[0].image.save(d)

c = C()
c.run()
