import torch
import torch.nn as nn
import sys
import os

# setup.py로 설치했으므로 Gabor CNN 모듈을 직접 임포트합니다.
from gcn.layers.GConv import GConv

# __all__은 이 파일에서 어떤 함수/클래스를 외부로 공개할지 정의합니다.
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# ARGS를 인자로 받도록 수정
def conv3x3(in_planes, out_planes, M, nScale, args, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return GConv(in_planes // M, out_planes // M, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                 bias=False, dilation=dilation, M=args.GConv_M, nScale=nScale)

# ARGS를 인자로 받도록 수정
def conv1x1(in_planes, out_planes, M, nScale, args, stride=1):
    """1x1 convolution"""
    return GConv(in_planes // M, out_planes // M, kernel_size=1, stride=stride,
                 bias=False, M=args.GConv_M, nScale=nScale)


class BasicBlock(nn.Module):
    expansion = 1

    # ARGS를 인자로 받도록 수정
    def __init__(self, inplanes, planes, M, nScale, args, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.M = M
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # conv 함수 호출 시 args 전달
        self.conv1 = conv3x3(inplanes, planes, M, nScale, args, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, M, nScale, args)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    # ARGS를 인자로 받도록 수정
    def __init__(self, inplanes, planes, M, nScale, args, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # conv 함수 호출 시 args 전달
        self.conv1 = conv1x1(inplanes, width, M, nScale, args)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, M, nScale, args, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, M, nScale, args)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    
    # ARGS를 인자로 받도록 수정
    def __init__(self, block, layers, M, nScale, GCN_is_maxpool, args, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, input_channel=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.M = M
        self.nScale = nScale
        self.args = args # args 객체를 클래스 멤버로 저장
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = GConv(input_channel, self.inplanes // M, kernel_size=5, stride=2,
                           padding=2, bias=False, expand=True, M=M, nScale=nScale)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if GCN_is_maxpool else nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0], M, nScale)
        self.layer2 = self._make_layer(block, 128, layers[1], M, nScale, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], M, nScale, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], M, nScale, stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion // M, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, M, nScale, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, M, nScale, self.args, stride), # self.args 전달
                norm_layer(planes * block.expansion),
            )

        layers = []
        # BasicBlock 생성자에 self.args 전달
        layers.append(block(self.inplanes, planes, M, nScale, self.args, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # BasicBlock 생성자에 self.args 전달
            layers.append(block(self.inplanes, planes, M, nScale, self.args, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x, bs, t = modify_input(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = torch.max(x.view(bs * t, -1, self.M), dim=2)[0]
        x = x.reshape(bs, t, -1)

        x = self.fc(x)
        x = torch.mean(x, dim=1)

        return x


def modify_input(input):
    if len(input.size()) == 6:
        assert input.shape[2] == 1, 'models.py/modify_input()'
        input = input.squeeze(dim=2)
        bs, t, c, h, w = input.size()
        input = input.reshape(bs * t, c, h, w)
        return input, bs, t
    elif len(input.size()) == 4:
        bs = input.size()[0]
        return input, bs, 1
    else:
        assert False, 'modify_input()'


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # kwargs에 args가 포함되어 ResNet 클래스로 전달됨
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)