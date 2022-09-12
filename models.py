import re
import torch
import timm
from torch import nn
from torchvision import transforms, models

class EffNet(nn.Module):
    def __init__(self, name='b0', num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.effnet = timm.create_model(f'efficientnet_{name}', pretrained=True)
        self.effnet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x =  self.effnet(x)
        if self.num_classes > 1:
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    e = EffNet('b0')
    x = torch.rand([3, 3, 256, 256])
    y = e(x)
    print(y)
