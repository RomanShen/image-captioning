import torch
from .modules import Classifier
from .modules import MyVgg16
from .modules import VAEDecoder
from torch import nn


class VggVAE(nn.Module):
    def __init__(self, pretrained=True):
        super(VggVAE, self).__init__()
        self.vgg = MyVgg16(pretrained=pretrained)
        self.classifier = Classifier()
        self.vae_decoder = VAEDecoder()

    def forward(self, x):
        z, features = self.vgg(x)
        x_recon, mu, logvar = self.vae_decoder(z)
        features = self.classifier(features)
        return features, x_recon, mu, logvar


if __name__ == '__main__':
    # import sys
    # print(sys.modules)
    vgg = VggVAE(pretrained=False)
    output = vgg(torch.rand(4, 3, 224, 224))
    for i in output:
        print(i.shape)