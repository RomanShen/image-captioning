import torch
from .modules import Classifier
from .modules import MyVgg16
from torch import nn


class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        self.vgg = MyVgg16(pretrained=pretrained)
        self.classifier = Classifier()

    def forward(self, x):
        return self.classifier(self.vgg(x)[1])


if __name__ == '__main__':
    import sys
    print(sys.modules)
    vgg = Vgg16()
    output = vgg(torch.rand(4, 3, 224, 224))
    print(output.shape)