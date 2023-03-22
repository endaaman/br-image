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

from endaaman import Commander, with_wrote, with_log, pad_to_size


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

    'C_hor3': ROIRule( # (1280, 960)
        (64, 141),
        (704, 141),
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

def get_res_code_by_image(img):
    if img.size == (960, 720):
        t = 'A'
    elif img.size == (1280, 720):
        t = 'B'
    elif img.size == (1280, 960):
        t = 'C'
    elif img.size == (1552, 873):
        t = 'D'
    else:
        t = 'unknown'
    return t


class C(Commander):
    def arg_convert_dicom(self, parser):
        parser.add_argument('--src', default='data/DICOM')
        parser.add_argument('--dest', default='data/images')

    def run_convert_dicom(self):
        os.makedirs(self.args.dest, exist_ok=True)
        pattern = os.path.join(self.args.src, '*/*')
        paths = sorted(glob(pattern, recursive=True))
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
            t = get_res_code_by_image(img)
            data.append({
                'id': id_,
                'style': t,
            })

        pd.DataFrame(data).to_csv(with_wrote('out/types.csv'), index=False)


    def run_check_integrity(self):
        df = pd.read_excel('data/label.xlsx', index_col=0)

        target_dirs = [
            'images',
            'crop/p',
            'crop/e',
            'crop/m',
            'crop/pe',
            'crop/pem',
        ]

        for td in target_dirs:
            print(f'checking {td}')
            paths = sorted(glob(os.path.join('data', td, '/*.png')))
            for p in paths:
                name = os.path.splitext(os.path.basename(p))[0]
                if not name in df.index:
                    print(f'{p} exists but {name} is not registered to df')

        print('checking pe')
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            if not name in df.index:
                print(f'{p} exists but {name} is not registered to df')

        print('checking df')
        for idx, row in df.iterrows():

            for td in target_dirs:
                s = td.split('/')
                fn = f'{idx}.png' if len(s) == 1 else f'{idx}_{s[-1]}.png'
                path = os.path.join('data', td, fn)

                if not os.path.isfile(path):
                    print(f'{idx} is not registered but {path} does not exist')

            if pd.isna(row.resolution):
                i = Image.open(f'data/images/{idx}.png')
                print(f'{idx} has no res code. It may be: ', get_res_code_by_image(i))

        print('done')

    def arg_crop(self, parser):
        parser.add_argument('--src', default='data/images/')
        parser.add_argument('--dest', default='data/crop/')
        parser.add_argument('--target', '-t', type=str, nargs='+', default=[])
        parser.add_argument('--swap', action='store_true')
        parser.add_argument('--square', action='store_true')
        parser.add_argument('--id', type=str)

    def run_crop(self):
        df = pd.read_excel('data/label.xlsx', index_col=0)
        os.makedirs(self.args.dest, exist_ok=True)

        if len(self.args.target) > 0:
            df = df.loc[self.args.target]

        total = len(df)
        for i, (idx, row) in (t:=tqdm(enumerate(df.iterrows()), total=total)):
            if self.args.id and self.args.id != idx:
                continue
            rule_code = f'{row.resolution}_{row.position}'
            rule = roi_rules.get(rule_code, None)
            if not rule:
                raise ValueError(f'{idx}({rule_code}) is unknown image type.')

            original_image_path = os.path.join(self.args.src, f'{idx}.png')
            original_image = Image.open(original_image_path)

            plain_image = original_image.crop(rule.p_rect()).convert('L')
            enhance_image = original_image.crop(rule.e_rect()).convert('L')
            assert enhance_image.size == plain_image.size
            base_size = plain_image.size
            if row['swap'] ^ self.args.swap:
                enhance_image, plain_image = plain_image, enhance_image

            e_arr = np.array(enhance_image)
            p_arr = np.array(plain_image)
            d_arr = np.zeros(p_arr.shape, dtype=np.uint8)

            mask_path = f'data/mask/{idx}.png'
            if os.path.isfile(mask_path):
                mask_image_base = Image.open(mask_path)
                dummy = np.zeros(base_size[::-1], dtype=np.uint8)
                rect = rule.e_rect() if row['swap'] else rule.p_rect()
                mask_image = mask_image_base.crop(rect)
                m_arr = np.array(mask_image)[:, :, 3]
            else:
                m_arr = d_arr

            # RGB -> GBR
            pem = Image.fromarray(np.stack([p_arr, e_arr, m_arr], 0).transpose((1, 2, 0)))
            pe =  Image.fromarray(np.stack([p_arr, e_arr, d_arr], 0).transpose((1, 2, 0)))
            m =  Image.fromarray(m_arr)
            p =  Image.fromarray(p_arr)
            e =  Image.fromarray(e_arr)

            targets = {
                'p': p,
                'e': e,
                'pe': pe,
                'm': m,
                'pem': pem,
            }

            for name, img in targets.items():
                img = pad_to_square(img, 720)
                d = os.path.join(self.args.dest, name)
                os.makedirs(d, exist_ok=True)
                img.save(os.path.join(d, f'{idx}_{name}.png'))

            t.set_description(f'{idx} {i}/{total}')
            t.refresh()


    def arg_extract_mask(self, parser):
        parser.add_argument('--src', '-s', default='data/draw')
        parser.add_argument('--dest', '-d', default='data/mask')

    def run_extract_mask(self):
        if os.path.isdir(self.args.src):
            paths = sorted(glob(os.path.join(self.args.src, '*.xcf')))
        elif os.path.isfile(self.args.src):
            paths = [self.args.src]
        else:
            raise RuntimeError(f'Invalid src: {self.args.src}')

        os.makedirs(self.args.dest, exist_ok=True)

        for p in tqdm(paths):
            name = os.path.splitext(os.path.basename(p))[0]
            prj = GimpDocument(p)
            d = os.path.join(self.args.dest, f'{name}.png')
            prj.layers[0].image.save(d)


    def run_drop_ba(self):
        df = pd.read_excel('data/label.xlsx', index_col=0)
        df = df.reset_index()

        df['id'] = -1
        for idx, row in df.iterrows():
            m = re.match(r'^.*(\d\d\d)_\d$', row['name'])
            if not m:
                raise RuntimeError('Invalid row:', idx, row)
            df.loc[idx, 'id'] = int(m[1])

        df.duplicated(keep='last')
        df.loc[df['id'].duplicated(keep='first'), 'id'] = -1

        df.to_excel(with_wrote('data/label_new.xlsx'), index=False)


c = C()
c.run()
