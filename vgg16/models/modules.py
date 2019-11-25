import torch
from torch import nn
from torchvision.models import vgg16_bn


class MyVgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(MyVgg16, self).__init__()
        vgg16 = vgg16_bn(pretrained=pretrained)
        self.features = vgg16.features
        # self.avgpool = vgg16.avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = vgg16.classifier[:6]

    def forward(self, x):
        x = self.features(x)
        y = self.avgpool(x)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return x, y


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 45),
            nn.LogSoftmax(dim=1)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VAEDecoder(nn.Module):
    def __init__(self):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)
        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, 512 * 7 * 7),
            nn.BatchNorm1d(512 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.decode = make_decoder_layers()
        self._initialize_weights()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc4(self.fc3(z)).view(-1, 512, 7, 7)
        return self.decode(z), mu, logvar

    def encode(self, x):
        x = x.view(-1, 7 * 7 * 512)
        x = self.fc1(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_decoder_layers():
    cfg = ['M', 512, 512, 512, 'M', 512, 512, 256, 'M', 256, 256, 128, 'M', 128, 64, 'M', 64, 3]
    layers = []
    in_channels = 512
    for v in cfg:
        if v == 'M':
            layers += [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)]
        else:
            convtran2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
            layers += [convtran2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    model = VAEDecoder()
    output = model(torch.rand(4, 512, 7, 7))
    print(output[0].shape)


