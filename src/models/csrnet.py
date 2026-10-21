"""CSRNet (Li et al., CVPR 2018): VGG16 frontend + dilated-convolution backend.

Frontend : first 10 conv layers of VGG16 (3 max-pools), pretrained on ImageNet.
Backend  : six 3x3 convs with dilation 2 (512,512,512,256,128,64) + 1x1 conv.
Output   : density map at 1/8 of the input size; its sum is the people count.
"""
import torch.nn as nn
from torchvision import models


def _make_frontend(pretrained):
    try:
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)
    except AttributeError:  # older torchvision
        vgg = models.vgg16(pretrained=pretrained)
    return nn.Sequential(*list(vgg.features.children())[:23])


def _make_backend(cfg=(512, 512, 512, 256, 128, 64), in_channels=512):
    layers = []
    for v in cfg:
        layers += [nn.Conv2d(in_channels, v, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, pretrained=True, freeze_frontend=False):
        super().__init__()
        self.frontend = _make_frontend(pretrained)
        self.backend = _make_backend()
        self.output_layer = nn.Conv2d(64, 1, 1)
        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
        if freeze_frontend:  # Experiment: freeze vs fine-tune
            for p in self.frontend.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.output_layer(self.backend(self.frontend(x)))
