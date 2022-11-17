import re
import torch
import timm
from torch import nn
from torchvision import transforms, models

class EffNet(nn.Module):
    def __init__(self, depth=0, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(f'tf_efficientnetv2_b{depth}', pretrained=True)
        c = self.base.classifier.in_features
        self.base.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        x =  self.base(x)
        if self.num_classes > 1:
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x


class VGG(nn.Module):
    def __init__(self, name, num_classes=1, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        vgg = getattr(models, name)
        if not re.match(r'^vgg', name):
            raise Exception(f'At least starting with "vgg": {name}')
        if not vgg:
            raise Exception(f'Invalid model name: {name}')

        base = vgg(pretrained=pretrained)
        self.convs = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.num_classes > 1:
            x = nn.functional.softmax(x)
        else:
            x = torch.sigmoid(x)
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x


def create_model(name):
    if re.match(r'^vgg', name):
        return VGG(name=name), 256

    m = re.match(r'^eff_b([0-4])$', name)
    if m:
        return EffNet(depth=m[1]), 512

    raise ValueError(f'Invalid name: {name}')

if __name__ == '__main__':
    # m = EffNet('b0')
    model, _ = create_model('vgg16_bn')
    x = torch.rand([3, 3, 256, 256])
    y = model(x)
    print(y)
