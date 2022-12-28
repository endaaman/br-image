import re
from typing import NamedTuple

import torch
from torch import nn
from torchvision import transforms, models
import timm
import segmentation_models_pytorch as smp

from endaaman import Commander, curry as c


class TimmModel(nn.Module):
    def __init__(self, name='tf_efficientnetv2_b0', num_classes=1, activation=True):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.base = timm.create_model(name, pretrained=True, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x, activate=True):
        x = self.base(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x


def create_model(s):
    return TimmModel(name=s, num_classes=1)


SMP_ARGS = dict(in_channels=3, activation='sigmoid', classes=1, encoder_weights=None)

SMP = {
    # 'unet11': c(UNet11, num_classes=1, pretrained=pretrained),
    # 'unet16': c(UNet16, num_classes=1, pretrained=pretrained),
    **{
        f'seg_unet1{w}': c(smp.Unet, f'vgg1{w}_bn',  **SMP_ARGS) for w in [1, 3, 6, 9]
    },
    **{
        f'seg_eff_b{w}': c(smp.Unet, f'efficientnet-b{w}', **SMP_ARGS) for w in range(8)
    },
    'seg_deeplab': c(smp.DeepLabV3Plus, **SMP_ARGS),
    'seg_mnet_v2': c(smp.Unet, 'mobilenet_v2', **SMP_ARGS),
    'seg_mnet_v3l': c(smp.Unet, 'timm-mobilenetv3_large_100', **SMP_ARGS),
    'seg_mnet_v3s': c(smp.Unet, 'timm-mobilenetv3_small_100', **SMP_ARGS),
}

def create_seg_model(name):
    return SMP[name]()



class CMD(Commander):
    def run_a(self):
        model = create_model('tf_efficientnetv2_b0')
        count = sum(p.numel() for p in model.parameters()) / 1_000_000
        print(f'count: {count:.2f}M')
        x = torch.rand([2, 3, 512, 512])
        y = model(x)
        # loss = CrossEntropyLoss()
        # print('y', y, y.shape, 'loss', loss(y, torch.LongTensor([1, 1])))
        print('y', y.shape)


    def run_b(self):
        m = create_seg_model('seg_eff_b0')
        x = torch.rand([2, 3, 512, 512])
        y = m(x)
        # loss = CrossEntropyLoss()
        # print('y', y, y.shape, 'loss', loss(y, torch.LongTensor([1, 1])))
        print('y', y.shape)


if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
