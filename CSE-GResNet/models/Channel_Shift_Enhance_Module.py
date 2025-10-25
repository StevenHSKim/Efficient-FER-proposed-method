import torch
import torch.nn as nn
import math
import numpy as np

# ARGS를 직접 임포트하는 대신, 함수가 args를 인자로 받도록 수정
def get_hw_layer(args):
    if args.model == 'GCN' and not args.GCN_is_maxpool:
        return {
            'layer1': [56, 56, 64], 'layer2': [28, 28, 128],
            'layer3': [14, 14, 256], 'layer4': [7, 7, 512],
        }
    else:
        return {
            'layer1': [28, 28, 64], 'layer2': [14, 14, 128],
            'layer3': [7, 7, 256], 'layer4': [4, 4, 512],
        }

# 클래스와 함수들이 args 객체를 전달받도록 수정
class ChannelEnhance_4(nn.Module):
    def __init__(self, net, h, w, input_channels, n_div_s, block_position_s, args):
        super(ChannelEnhance_4, self).__init__()
        self.net = net
        self.args = args
        M = 4 if self.args.model == 'GCN' else 1
        self.fold_s = M * h * w // n_div_s
        self.fold_e = M * h * w // self.args.TSM_div_e

        self.conv_s = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=3, padding=1, groups=M * h * w, bias=False)
        self.conv_s, self.i_s = init_conv(self.conv_s, n_div_s, M * h * w, block_position_s, self.fold_s, self.args, init_strategy=3, kernelsize=3)

        self.conv_e = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=self.args.channel_enhance_kernelsize,
                                padding=math.floor(self.args.channel_enhance_kernelsize / 2), groups=M * h * w, bias=False)
        self.conv_e, self.i_e = init_conv(self.conv_e, self.args.TSM_div_e, M * h * w, self.args.channel_block_position_e, self.fold_e, self.args,
                                          init_strategy=self.args.channel_enhance_init_strategy, kernelsize=self.args.channel_enhance_kernelsize, channel_enhance=2)

    def forward(self, x):
        x = enhance(x, [self.conv_s, self.conv_e], [self.i_s, self.i_e], [self.fold_s, self.fold_e], self.args)
        return self.net(x)

def residual_conv(x, conv, i, fold, channel_block_position):
    identity = x
    x = conv(x)
    if channel_block_position == 'stochastic':
        x[:, i, :] = identity[:, i, :] + x[:, i, :]
    else:
        x[:, i * fold:(i + 2) * fold, :] = identity[:, i * fold:(i + 2) * fold, :] + x[:, i * fold:(i + 2) * fold, :]
    return x

def fusion_SE(x, conv, i, fold, args):
    conv_s, conv_e = conv[0], conv[1]
    i_e = i[1]
    fold_e = fold[1]
    if args.fusion_SE == 'A':
        x = conv_s(x)
        x = residual_conv(x, conv_e, i_e, fold_e, args.channel_block_position_e) if args.TSM_conv_insert_e == 'residual' else conv_e(x)
    elif args.fusion_SE == 'B':
        x_t = x
        x = conv_s(x)
        x_t = residual_conv(x_t, conv_e, i_e, fold_e, args.channel_block_position_e) if args.TSM_conv_insert_e == 'residual' else conv_e(x_t)
        x = x + x_t
    elif args.fusion_SE == 'C':
        x = conv_s(x)
        x_t = residual_conv(x, conv_e, i_e, fold_e, args.channel_block_position_e) if args.TSM_conv_insert_e == 'residual' else conv_e(x)
        x = x + x_t
    else:
        assert False, 'fusion_SE()!!!'
    return x

def enhance(x, conv, i, fold, args):
    bs, c, h, w = x.size()
    x = x.reshape(bs, c // 4, 4 * h * w) if args.model == 'GCN' else x.reshape(bs, c, h * w)
    x = x.permute([0, 2, 1])
    if args.TSM_channel_enhance in [4]:
        x = fusion_SE(x, conv, i, fold, args)
    else:
        x = residual_conv(x, conv, i, fold, args.channel_block_position) if args.TSM_conv_insert_e == 'residual' else conv(x)
    x = x.permute([0, 2, 1])
    x = x.reshape(bs, c, h, w)
    return x

def make_channel_shift(net, args):
    assert args.base_model in ['resnet18', 'resnet34'], 'make_temporal_shift()!!!'
    HW_Layer = get_hw_layer(args)
    for position in args.TSM_position:
        layer = getattr(net, position)
        layer = make_BasicBlock_shift_test(layer, HW_Layer[position], args)
        setattr(net, position, layer)
    return net

def make_BasicBlock_shift_test(stage, hwc, args, mode='residual'):
    assert mode in ['inplace', 'residual'], 'make_BasicBlock_shift_test()!!!'
    blocks = list(stage.children())
    modified_module = blocks[-1] if mode == 'inplace' else blocks[-1].conv1
    
    # 다른 ChannelEnhance 버전이 필요하면 여기에 추가
    if args.TSM_channel_enhance == 4:
        modified_module = ChannelEnhance_4(modified_module, hwc[0], hwc[1], hwc[2],
                                           n_div_s=args.TSM_div, block_position_s=args.channel_block_position,
                                           args=args)
    else:
        raise NotImplementedError(f"TSM_channel_enhance version {args.TSM_channel_enhance} not implemented")

    if mode == 'inplace':
        blocks[-1] = modified_module
    else:
        blocks[-1].conv1 = modified_module
    return nn.Sequential(*blocks)

def init_conv(conv, n_div, hw, block_position, fold, args, init_strategy=3, kernelsize=3, channel_enhance=3):
    if channel_enhance == 3:
        conv.weight.requires_grad = False
    nn.init.constant_(conv.weight, 0)
    if block_position == 'start': i = 0
    elif block_position == 'middle': i = int(n_div // 2 - 1)
    elif block_position == 'end': i = n_div - 2
    elif block_position == 'stochastic':
        i = np.random.choice(np.arange(hw), 2 * fold, replace=False)
    else:
        assert False, 'init_conv()!!!'
    if block_position == 'stochastic':
        conv = init_conv_stochastic(conv, i, hw, fold, init_strategy, kernelsize)
    else:
        conv = init_conv_for_others(conv, i, hw, fold, init_strategy, kernelsize)
    return conv, i

def init_conv_for_others(conv, i, hw, fold, init_strategy, kernelsize):
    middle_i = math.floor(kernelsize / 2)
    if i * fold != 0: conv.weight.data[:i * fold, 0, middle_i] = 1
    if (i + 2) * fold < hw: conv.weight.data[(i + 2) * fold:, 0, middle_i] = 1
    if init_strategy == 1: conv.weight.data[i * fold:(i + 2) * fold, 0, :] = 1
    elif init_strategy == 2: nn.init.kaiming_uniform_(conv.weight.data[i * fold:(i + 2) * fold, 0, :])
    elif init_strategy == 3:
        conv.weight.data[i * fold:(i + 1) * fold, 0, middle_i + 1] = 1
        conv.weight.data[(i + 1) * fold: (i + 2) * fold, 0, middle_i - 1] = 1
    return conv

def init_conv_stochastic(conv, i, hw, fold, init_strategy, kernelsize):
    middle_i = math.floor(kernelsize / 2)
    conv.weight.data[[k for k in set(range(hw)) - set(i)], 0, middle_i] = 1
    if init_strategy == 1: conv.weight.data[i, 0, :] = 1
    elif init_strategy == 2:
        tem = conv.weight.data[i, 0, :]
        nn.init.kaiming_uniform_(tem)
        for k, t in zip(i, tem): conv.weight.data[k, 0, :] = t
    elif init_strategy == 3:
        conv.weight.data[i[:fold], 0, middle_i + 1] = 1
        conv.weight.data[i[fold:], 0, middle_i - 1] = 1
    return conv