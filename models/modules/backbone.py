import torch
import torch.nn as nn
from collections import OrderedDict
from .resnet_plop import ResNet as ResNet_plop
from .resent_official import ResNet as ResNet_official

__all__ = ["ResNet101"]


def ResNet101(norm_act=nn.BatchNorm2d, norm_name='bn_sync', output_stride=16, pretrained=True):
    if norm_name == 'iabn_sync':

        model = ResNet_plop(
            structure=[3, 4, 23, 3],
            bottleneck=True,
            norm_act=norm_act,
            output_stride=16
        )
        if pretrained:

            model._load_pretrained_model('weight/pretrained/imagenet/resnet/resnet101_iabn_sync.pth.tar')

    elif norm_name == 'bn_sync':
        model = ResNet_official(
            structure=[3, 4, 23, 3],
            bottleneck=True,
            norm_act=norm_act,
            output_stride=16
        )
        if pretrained:
            model._load_pretrained_model('https://download.pytorch.org/models/resnet101-63fe2227.pth')
    else:
        raise NotImplementedError

    return model
