import re

import torch
from torch import nn
from torchvision import transforms, models
import timm
import segmentation_models_pytorch as smp


class EffNet(nn.Module):
    def __init__(self, name='v2_b0', num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        if m := re.match('^v2_(.+)$', name):
            model_name = f'tf_efficientnetv2_{m[1]}'
        else:
            model_name = f'tf_efficientnet_{name}'

        self.base = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x):
        x =  self.base(x)
        if self.num_classes > 1:
            x = torch.softmax(x, dim=-1)
        else:
            x = torch.sigmoid(x)
        return x


def create_model(name, num_classes):
    if m := re.match(r'^eff_(b[0-7])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(b[0-7]_ns)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_b[0-4])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_s|m|l)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    raise ValueError(f'Invalid name: {name}')


def create_seg_model(name, num_classes):
    if m := re.match(r'^eff_(b[0-7])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(b[0-7]_ns)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_b[0-4])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_s|m|l)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    raise ValueError(f'Invalid name: {name}')


if __name__ == '__main__':
    n = 'eff_v2_b3'
    model = create_model(n, 3)
    count = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f'count: {count:.2f}M')
    x_ = torch.rand([2, 3, 512, 512])
    y_ = model(x_)
    # loss = CrossEntropyLoss()
    # print('y', y, y.shape, 'loss', loss(y, torch.LongTensor([1, 1])))
    print('y', y_, y_.shape)
