from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from models.resnet_torch import *
from models.resnet import MoCo
from models.saliency_sampler import Saliency_Sampler
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

# from ..utils.serialization import load_checkpoint, copy_state_dict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class UnNormalize:
    #restore from T.Normalize
    
    def __init__(self,device,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
        self.mean=torch.tensor(mean).view((1,-1,1,1)).to(device)
        self.std=torch.tensor(std).view((1,-1,1,1)).to(device)
    def __call__(self,x):
        x=(x*self.std)+self.mean
        return torch.clamp(x,0,None)


class MSCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei

class AFFM(nn.Module):
    def __init__(self, channel=64):
        super(AFFM, self).__init__()

        self.msca = MSCA()

    def forward(self, x, y):

        xy = x + y
        wei = self.msca(xy)
        xo = x * wei + y * (1 - wei)

        return xo



class Baseline(nn.Module):
    def __init__(self, args, code_length=12, num_classes=200, pretrained=True):
        super(Baseline, self).__init__()
        self.backbone_global = resnet50(islocal=False, pretrained=pretrained, num_classes=num_classes)
        self.backbone_local = resnet50(islocal=True, pretrained=pretrained, num_classes=num_classes)
        self.sample = Saliency_Sampler(args.device, self.backbone_global, 1024, 224, 224)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, code_length)
        self.affm = AFFM()
        self.act2 = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.unNormalize = UnNormalize(args.device)


    def forward(self, x):

        x_orginal = self.unNormalize(x)
        room_fea, orginal_fea, sample_map = self.sample(x)
        x_room = self.unNormalize(room_fea)
        room_fea, room_midfea = self.backbone_global(room_fea)
        hash_code = self.affm(orginal_fea, room_fea)
        # hash_code = orginal_fea+room_fea
        hash_code = self.avgpool(hash_code)
        hash_code = torch.flatten(hash_code, 1)
        hash_code = self.fc(hash_code)

        return hash_code, x_orginal, x_room, sample_map


def baseline(args, code_length, num_classes, pretrained=False,  progress=True, **kwargs):
    model = Baseline(args, code_length, num_classes, pretrained, **kwargs)
    return model
