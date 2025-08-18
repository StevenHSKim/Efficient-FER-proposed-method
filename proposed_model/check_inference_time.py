import os
import sys
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import ShuffleSplit, train_test_split
import random
import time
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 import - 수정된 import
# 두 번째 파일의 모델을 사용 (BetterShuffleNet_ChannelAttention)
import torch.nn.functional as F
import torch.nn.init as init
import math

# 모델 정의를 직접 포함 (두 번째 파일의 내용)
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SpatialGlance(nn.Module):
    """Spatial Glance module for facial expression recognition"""
    def __init__(self, in_channels):
        super(SpatialGlance, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 7, 1, 3, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            hswish(),
            nn.Conv2d(in_channels // 8, 1, 7, 1, 3, bias=False),
            hsigmoid()
        )
    
    def forward(self, x):
        attention = self.spatial_attention(x)
        return x * attention

class MDConv(nn.Module):
    """Mixed Depthwise Convolution with 3 groups using dilated 3x3 convolutions"""
    def __init__(self, channels, stride=1):
        super(MDConv, self).__init__()
        self.channels = channels
        self.stride = stride
        
        # Handle channels not divisible by 3
        base_channels = channels // 3
        remainder = channels % 3
        
        # Distribute remainder channels to first few groups
        self.group_channels = []
        for i in range(3):
            if i < remainder:
                self.group_channels.append(base_channels + 1)
            else:
                self.group_channels.append(base_channels)
        
        # Different dilation rates for 3x3 kernel: dilation 1, 2, 3 (effective 3x3, 5x5, 7x7)
        self.conv3x3_d1 = nn.Conv2d(self.group_channels[0], self.group_channels[0], 
                                   kernel_size=3, stride=stride, padding=1, dilation=1,
                                   groups=self.group_channels[0], bias=False)
        self.conv3x3_d2 = nn.Conv2d(self.group_channels[1], self.group_channels[1], 
                                   kernel_size=3, stride=stride, padding=2, dilation=2,
                                   groups=self.group_channels[1], bias=False)
        self.conv3x3_d3 = nn.Conv2d(self.group_channels[2], self.group_channels[2], 
                                   kernel_size=3, stride=stride, padding=3, dilation=3,
                                   groups=self.group_channels[2], bias=False)
    
    def forward(self, x):
        # Split input into 3 groups with possibly different sizes
        x_splits = torch.split(x, self.group_channels, dim=1)
        
        # Apply different dilation rates (effective 3x3, 5x5, 7x7 receptive fields)
        out1 = self.conv3x3_d1(x_splits[0])  # dilation=1 (3x3 effective)
        out2 = self.conv3x3_d2(x_splits[1])  # dilation=2 (5x5 effective)
        out3 = self.conv3x3_d3(x_splits[2])  # dilation=3 (7x7 effective)
        
        # Concatenate results
        return torch.cat([out1, out2, out3], dim=1)

class SELayer(nn.Module):
    """SE Layer following first method - applied to full output channels"""
    def __init__(self, inplanes, isTensor=True):
        super(SELayer, self).__init__()
        if isTensor:
            # if the input is (N, C, H, W)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
            )
        else:
            # if the input is (N, C)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Linear(inplanes, inplanes // 4, bias=False),
                nn.BatchNorm1d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes // 4, inplanes, bias=False),
            )

    def forward(self, x):
        atten = self.SE_opr(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten

def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x

def conv_1x1_bn(in_c, out_c, stride=1, nonlinear=None):
    """1x1 conv with optional nonlinear activation"""
    if nonlinear is None:
        nonlinear = hswish()
    
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_c),
        nonlinear
    )

class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, nonlinear, stage_idx, downsample=False, use_mdconv=False, use_residual=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        self.use_mdconv = use_mdconv
        self.use_residual = use_residual
        self.stage_idx = stage_idx
        half_c = out_c // 2
        
        # SE 적용 조건: stage 2 이후 (stage_idx >= 2)
        self.use_se = stage_idx >= 2
        
        if downsample:
            # Stride=2 downsample 블록
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nonlinear
            )
            
            if use_mdconv:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(half_c),
                    nonlinear,
                    MDConv(half_c, stride=2),
                    nn.BatchNorm2d(half_c),
                    nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(half_c),
                    nonlinear
                )
            else:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(half_c),
                    nonlinear,
                    nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                    nn.BatchNorm2d(half_c),
                    nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(half_c),
                    nonlinear
                )
            
            # SE layer for branch2 output (first method: at the end)
            if self.use_se:
                self.se_layer = SELayer(half_c)
                
        else:
            # Stride=1 기본 블록
            assert in_c == out_c
            
            # 공통 레이어들
            self.branch2_conv1 = nn.Sequential(
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nonlinear
            )
            
            if use_mdconv:
                self.branch2_mdconv = MDConv(half_c, stride=1)
                self.branch2_bn = nn.BatchNorm2d(half_c)
            else:
                self.branch2_dwconv = nn.Sequential(
                    nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                    nn.BatchNorm2d(half_c)
                )
            
            self.branch2_conv2 = nn.Sequential(
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nonlinear
            )
            
            # SE layer for branch2 output (first method: at the end)
            if self.use_se:
                self.se_layer = SELayer(half_c)

    def forward(self, x):
        if self.downsample:
            # Stride=2 블록
            branch1_out = self.branch1(x)
            branch2_out = self.branch2(x)
            
            # SE 적용 (첫 번째 방법: 마지막에 적용)
            if self.use_se:
                branch2_out = self.se_layer(branch2_out)
                
            out = torch.cat((branch1_out, branch2_out), 1)
        else:
            # Stride=1 블록
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]  # 좌측 분기 (identity)
            x2 = x[:, c:, :, :]  # 우측 분기 (변환)
            
            # 우측 분기 변환: 1×1 Conv + BN + ReLU
            out2 = self.branch2_conv1(x2)
            
            # DW Conv (MDConv 또는 일반 3x3)
            if self.use_mdconv:
                out2 = self.branch2_mdconv(out2)
                out2 = self.branch2_bn(out2)
            else:
                out2 = self.branch2_dwconv(out2)
            
            # 1×1 Conv + BN + ReLU
            out2 = self.branch2_conv2(out2)
            
            # Residual connection (옵션)
            if self.use_residual:
                out2 = out2 + x2
            
            # SE 적용 (첫 번째 방법: 마지막에 적용)
            if self.use_se:
                out2 = self.se_layer(out2)
            
            # Concat and shuffle
            out = torch.cat((x1, out2), 1)
            out = channel_shuffle(out, 2)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, input_channels=512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(input_channels, input_channels // 16),
            nn.BatchNorm1d(input_channels // 16),
            hswish(),
            nn.Linear(input_channels // 16, input_channels),
            hsigmoid()
        )
    
    def forward(self, x):
        # Global Average Pooling
        features = self.gap(x)
        features = features.view(features.size(0), -1)
        
        # Channel attention weights
        y = self.attention(features)
        
        # Apply attention weights to original features
        out = features * y
        
        return out

class ChannelAttentionHead(nn.Module):
    def __init__(self, input_channels=512):
        super().__init__()
        self.channel_attention = ChannelAttention(input_channels)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, x):
        out = self.channel_attention(x)
        return out

class BetterShuffleNet(nn.Module):
    def __init__(self, input_size=224, n_class=7, model_size='1.0x', embedding_size=136, 
                 use_mdconv=True, custom_stage_repeats=None, use_residual=False):
        super(BetterShuffleNet, self).__init__()
        print(f'Model size is {model_size}, Input size is {input_size}, Use MDConv: {use_mdconv}, Use Residual: {use_residual}')
        print("Using first method SE application (stage 2+ only)")
        
        # 사용자 정의 stage repeat 설정 지원
        if custom_stage_repeats is not None:
            self.stage_repeat_num = custom_stage_repeats
            print(f"Using custom stage repeats: {custom_stage_repeats}")
        else:
            # 입력 크기에 따른 적응적 stage repeat 설정
            if input_size <= 112:
                self.stage_repeat_num = [3, 6, 3]  # 총 12 블록
                print("Using reduced stage repeats [3, 6, 3] for small input size")
            elif input_size <= 160:
                self.stage_repeat_num = [3, 6, 4]  # 총 13 블록  
                print("Using medium stage repeats [3, 6, 4] for medium input size")
            else:
                self.stage_repeat_num = [4, 8, 4]  # 총 16 블록
                print("Using standard stage repeats [4, 8, 4] for large input size")
        
        if input_size not in [112, 128, 160, 192, 224, 256, 288]:
            raise ValueError(f"Input size {input_size} not supported. Use one of [112, 128, 160, 192, 224, 256, 288]")
        
        self.model_size = model_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.use_mdconv = use_mdconv
        self.use_residual = use_residual
        
        # 모델 크기에 따른 채널 수 설정
        if model_size == '0.5x':
            self.out_channels = [3, 24, 48, 96, 192, 512]
        elif model_size == '1.0x':
            self.out_channels = [3, 24, 116, 232, 464, 512]
        elif model_size == '1.5x':
            self.out_channels = [3, 24, 176, 352, 704, 512]
        elif model_size == '2.0x':
            self.out_channels = [3, 24, 244, 488, 976, 1024]
        else:
            raise NotImplementedError(f"Model size {model_size} not implemented")
            
        # First conv layer
        if input_size <= 112:
            self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 1, 1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(self.out_channels[1])
        self.relu = hswish()
        
        # Build stages dynamically
        in_c = self.out_channels[1]
        self.stages = nn.ModuleList()
        
        for stage_idx in range(len(self.stage_repeat_num)):
            out_c = self.out_channels[2 + stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]
            
            stage = self._make_stage(in_c, out_c, repeat_num, stage_idx, use_mdconv, use_residual)
            self.stages.append(stage)
            in_c = out_c
        
        # Spatial Glance modules - 특정 stage에만 적용
        if len(self.stages) >= 2:
            self.spatial_glance_s2 = SpatialGlance(self.out_channels[2])  # 첫 번째 stage 후
        if len(self.stages) >= 3:
            self.spatial_glance_s3 = SpatialGlance(self.out_channels[3])  # 두 번째 stage 후
        self.conv_last = conv_1x1_bn(self.out_channels[-2], self.out_channels[-1], 1, hswish())
        
        # Backbone final layer
        self.backbone_final = nn.Sequential(
            nn.Conv2d(self.out_channels[-1], self.embedding_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.embedding_size),
            hswish()
        )
        
        # Last SE layer (첫 번째 방법: 전체 네트워크 마지막에 적용)
        self.LastSE = SELayer(self.out_channels[-1])
        
        self._calculate_final_size()
        self._initialize_weights()
        
        # 디버깅 정보
        total_blocks = sum(self.stage_repeat_num)
        print(f"Total blocks: {total_blocks}")
        print(f"Stage repeats: {self.stage_repeat_num}")
        print(f"Number of stages: {len(self.stages)}")
        if use_residual:
            print(f"Residual connections enabled for stride=1 blocks")
        print(f"SE blocks applied from stage 2 onwards")
    
    def _make_stage(self, in_c, out_c, repeat_num, stage_idx, use_mdconv, use_residual):
        """Create a stage with specified number of blocks"""
        stages = []
        nonlinear = hswish()
        
        for i in range(repeat_num):
            if i == 0:
                # 첫 번째 블록 (downsample=True)
                stages.append(ShuffleBlock(in_c, out_c, nonlinear, stage_idx, 
                                         downsample=True, use_mdconv=False, 
                                         use_residual=False))  # downsample 블록은 residual 사용 안함
                in_c = out_c
            else:
                # 나머지 블록들 (stride=1)
                stages.append(ShuffleBlock(in_c, in_c, nonlinear, stage_idx, 
                                         downsample=False, use_mdconv=use_mdconv, 
                                         use_residual=use_residual))
        
        return nn.Sequential(*stages)

    def _calculate_final_size(self):
        """입력 크기에 따른 최종 feature map 크기 계산"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
            x = self.conv1(dummy_input)
            x = self.maxpool(x)
            
            for _ in range(len(self.stages)):
                x = F.avg_pool2d(x, 2)
            
            self.final_size = x.size(-1)
            print(f"Final feature map size: {self.final_size}x{self.final_size}")

    def forward(self, x):
        # Initial conv and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        stage_features = []
        
        # Process each stage
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            
            # Spatial Glance 적용 - 첫 번째와 두 번째 stage에만
            if stage_idx == 0 and hasattr(self, 'spatial_glance_s2'):
                # 첫 번째 stage 완료 후
                x = self.spatial_glance_s2(x)
                stage_features.append(x)
            elif stage_idx == 1 and hasattr(self, 'spatial_glance_s3'):
                # 두 번째 stage 완료 후  
                x = self.spatial_glance_s3(x)
                stage_features.append(x)
            else:
                # 다른 stage들
                stage_features.append(x)
        
        # Final conv
        conv_features = self.conv_last(x)
        
        # Last SE layer (첫 번째 방법: 글로벌 풀링 전에 적용)
        conv_features = self.LastSE(conv_features)
        
        # Backbone final output
        backbone_output = self.backbone_final(conv_features)
        
        # Return features (기존 코드와 호환성 유지)
        if len(stage_features) >= 2:
            return stage_features[0], stage_features[1], conv_features, backbone_output
        elif len(stage_features) == 1:
            return stage_features[0], stage_features[0], conv_features, backbone_output
        else:
            return x, x, conv_features, backbone_output

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ShuffleNetV2_ChannelAttention 별칭으로 추가 (기존 코드 호환성)
class ShuffleNetV2_ChannelAttention(nn.Module):
    """Complete model: ShuffleNet backbone + Channel Attention head (for compatibility)"""
    def __init__(self, input_size=224, n_class=7, model_size='1.0x', embedding_size=136):
        super(ShuffleNetV2_ChannelAttention, self).__init__()
        
        # BetterShuffleNet을 기반으로 구성
        self.backbone = BetterShuffleNet(input_size=input_size, n_class=n_class, 
                                       model_size=model_size, embedding_size=embedding_size,
                                       use_mdconv=True, use_residual=False)
        
        # Final conv layer
        input_channels = 1024 if model_size == '2.0x' else 512
        self.channel_attention_head = ChannelAttentionHead(input_channels=input_channels)
        
        # Final classification layers
        self.fc = nn.Linear(input_channels, n_class)
        self.bn = nn.BatchNorm1d(n_class)
    
    def forward(self, x):
        # Backbone forward
        s2_features, s3_features, conv_features, backbone_output = self.backbone(x)
        
        # Channel Attention Head forward
        attention_features = self.channel_attention_head(conv_features)
        
        # Final classification
        out = self.fc(attention_features)
        out = self.bn(out)
        
        # Return format 기존 코드와 호환
        return out, conv_features, attention_features

def warn(*args, **kwargs):
    pass
warnings.warn = warn

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser(description='ShuffleNetV2 Channel Attention Inference Speed Measurement')
    
    # 데이터셋 관련 설정 (학습 코드와 동일)
    parser.add_argument('--raf_path', type=str, 
                       default='./datasets/raf-basic', 
                       help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint_path', type=str,
                       default='./proposed_model/checkpoints/shufflenetv2_channel_attention_1.0x_iter1_epoch50_acc0.8500_bacc0.8200.pth',
                       help='Path to the checkpoint file')
    
    # 모델 파라미터 (학습 코드와 동일)
    parser.add_argument('--num_classes', type=int, default=7, help='Number of emotion classes.')
    parser.add_argument('--model_size', type=str, default='1.0x', 
                       choices=['0.5x', '1.0x', '1.5x', '2.0x'], 
                       help='ShuffleNetV2 model size')
    parser.add_argument('--embedding_size', type=int, default=136, help='Embedding size for backbone output.')
    
    # 실험 설정
    parser.add_argument('--gpu', type=str, default='0', help='GPU device number')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    # 데이터 분할 설정 (학습 코드와 동일)
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    parser.add_argument('--iteration', type=int, default=1, help='Which iteration to use for testing (1-based)')
    
    # 측정 관련 설정
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs for inference timing')
    parser.add_argument('--warm_up', type=int, default=3, help='Number of warm-up runs')
    parser.add_argument('--measure_detailed', action='store_true',
                       help='Measure detailed timing for each component')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to CSV and JSON files')
    
    # 추가 실험 설정
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16'],
                       help='Inference precision')
    parser.add_argument('--profile_memory', action='store_true',
                       help='Profile GPU memory usage')
    parser.add_argument('--create_visualizations', action='store_true', default=True,
                       help='Create timing and accuracy visualizations')
    
    return parser.parse_args()

def control_random_seed(seed):
    """랜덤 시드를 고정하는 함수 (학습 코드와 동일)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class RafDataSet(data.Dataset):
    """RAF-DB 데이터셋 로더 (학습 코드와 완전히 동일)"""
    def __init__(self, raf_path, phase, indices, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        label_file = os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
            
        df = pd.read_csv(label_file, sep=' ', header=None, names=['name', 'label'])
        self.file_names = df['name'].values[indices]
        self.labels = df['label'].values[indices] - 1

        self.file_paths = []
        for f in self.file_names:
            img_path = os.path.join(self.raf_path, 'Image/aligned', 
                                   f.split(".")[0] + "_aligned.jpg")
            self.file_paths.append(img_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
            
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def load_model_and_checkpoint(checkpoint_path, args, device):
    """모델 생성 및 체크포인트 로드 (학습 코드 구조와 동일)"""
    print(f"Creating ShuffleNetV2 Channel Attention model with size: {args.model_size}")
    
    # 모델 생성 (학습 코드와 완전히 동일한 파라미터)
    model = ShuffleNetV2_ChannelAttention(
        input_size=224, 
        n_class=args.num_classes, 
        model_size=args.model_size, 
        embedding_size=args.embedding_size
    )
    
    model.to(device)
    
    # 정밀도 설정
    if args.precision == 'fp16':
        model.half()
        print("Using FP16 precision")
    else:
        print("Using FP32 precision")
    
    # 체크포인트 로드
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        print("Using randomly initialized model")
        return model, 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 학습 코드의 저장 형식에 맞춰 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            accuracy = checkpoint.get('accuracy', 0)
            balanced_accuracy = checkpoint.get('balanced_accuracy', 0)
            
            print("Loaded model_state_dict from checkpoint")
            print(f"Checkpoint epoch: {epoch}")
            print(f"Checkpoint accuracy: {accuracy:.4f}")
            print(f"Checkpoint balanced accuracy: {balanced_accuracy:.4f}")
            
            return model, accuracy
        else:
            model.load_state_dict(checkpoint)
            print("Loaded state_dict directly from checkpoint")
            return model, 0
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized model")
        return model, 0

def measure_model_accuracy(model, test_loader, device, args):
    """모델 정확도 측정 (학습 코드의 test 함수와 유사한 구조)"""
    print("\nEvaluating model accuracy...")
    model.eval()
    
    running_loss = 0.0
    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
    
    y_true = []
    y_pred = []
    
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Accuracy evaluation")
        for imgs, targets in test_pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            if args.precision == 'fp16':
                imgs = imgs.half()
            
            # 학습 코드와 동일한 출력 형식
            out, features, attention_features = model(imgs)
            loss = criterion(out, targets)
            
            running_loss += loss.item()
            iter_cnt += 1
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)
            
            y_true.append(targets.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())
    
    running_loss = running_loss / iter_cnt
    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)
    
    return acc, balanced_acc, running_loss, y_true, y_pred

def measure_inference_time_basic(model, test_loader, device, args):
    """기본 추론 시간 측정"""
    print(f"\nMeasuring basic inference time over {args.num_runs} runs...")
    
    model.eval()
    batch_times = []
    
    with torch.no_grad():
        for run in range(args.num_runs):
            run_times = []
            
            for imgs, _ in tqdm(test_loader, desc=f"Run {run+1}/{args.num_runs}", leave=False):
                imgs = imgs.to(device)
                
                if args.precision == 'fp16':
                    imgs = imgs.half()
                
                # CUDA 이벤트를 사용한 정밀한 시간 측정
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize()
                start_event.record()
                
                # 학습 코드와 동일한 모델 호출 방식
                out, features, attention_features = model(imgs)
                
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_time = start_event.elapsed_time(end_event)
                run_times.append(elapsed_time)
            
            avg_time = sum(run_times) / len(run_times)
            batch_times.append(avg_time)
    
    return batch_times

def measure_inference_time_detailed(model, test_loader, device, args):
    """상세한 추론 시간 측정 (각 컴포넌트별)"""
    print(f"\nMeasuring detailed inference time...")
    
    model.eval()
    
    backbone_times = []
    attention_times = []
    classification_times = []
    total_times = []
    
    with torch.no_grad():
        for run in range(min(args.num_runs, 50)):  # 상세 측정은 최대 50회
            for imgs, _ in tqdm(test_loader, desc=f"Detailed Run {run+1}", leave=False):
                imgs = imgs.to(device)
                
                if args.precision == 'fp16':
                    imgs = imgs.half()
                
                # 전체 시간 측정
                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize()
                total_start.record()
                
                # Backbone 시간 측정
                backbone_start = torch.cuda.Event(enable_timing=True)
                backbone_end = torch.cuda.Event(enable_timing=True)
                
                backbone_start.record()
                s2_features, s3_features, conv_features, backbone_output = model.backbone(imgs)
                backbone_end.record()
                
                # Channel Attention 시간 측정
                attention_start = torch.cuda.Event(enable_timing=True)
                attention_end = torch.cuda.Event(enable_timing=True)
                
                attention_start.record()
                attention_features = model.channel_attention_head(conv_features)
                attention_end.record()
                
                # Classification 시간 측정
                cls_start = torch.cuda.Event(enable_timing=True)
                cls_end = torch.cuda.Event(enable_timing=True)
                
                cls_start.record()
                out = model.fc(attention_features)
                out = model.bn(out)
                cls_end.record()
                
                total_end.record()
                torch.cuda.synchronize()
                
                # 시간 계산
                backbone_time = backbone_start.elapsed_time(backbone_end)
                attention_time = attention_start.elapsed_time(attention_end)
                cls_time = cls_start.elapsed_time(cls_end)
                total_time = total_start.elapsed_time(total_end)
                
                backbone_times.append(backbone_time)
                attention_times.append(attention_time)
                classification_times.append(cls_time)
                total_times.append(total_time)
    
    return {
        'backbone': backbone_times,
        'attention': attention_times,
        'classification': classification_times,
        'total': total_times
    }

def measure_memory_usage(model, test_loader, device, args):
    """GPU 메모리 사용량 측정"""
    if not args.profile_memory:
        return {}
    
    print("\nMeasuring GPU memory usage...")
    
    model.eval()
    memory_stats = []
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_loader):
            if i >= 10:  # 처음 10개 배치만 측정
                break
                
            imgs = imgs.to(device)
            
            if args.precision == 'fp16':
                imgs = imgs.half()
            
            # 추론 전 메모리
            memory_before = torch.cuda.memory_allocated()
            
            out, features, attention_features = model(imgs)
            
            # 추론 후 메모리
            memory_after = torch.cuda.memory_allocated()
            memory_peak = torch.cuda.max_memory_allocated()
            
            memory_stats.append({
                'batch': i,
                'memory_before_mb': memory_before / 1024 / 1024,
                'memory_after_mb': memory_after / 1024 / 1024,
                'memory_peak_mb': memory_peak / 1024 / 1024,
                'memory_diff_mb': (memory_after - memory_before) / 1024 / 1024
            })
    
    return memory_stats

def create_confusion_matrix(y_true, y_pred, model_info, save_path):
    """Confusion Matrix 생성 및 저장 (학습 코드와 동일한 형식)"""
    cm = confusion_matrix(y_true, y_pred)
    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
               xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {model_info["accuracy"]:.4f}, Balanced Accuracy: {model_info["balanced_accuracy"]:.4f}')
    
    plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_timing_visualizations(results, save_path):
    """타이밍 결과 시각화"""
    if not results.get('create_visualizations', True):
        return
        
    print("\nCreating timing visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. 추론 시간 히스토그램
    plt.subplot(2, 3, 1)
    plt.hist(results['basic_times'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    plt.title('Inference Time Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    mean_time = np.mean(results['basic_times'])
    plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}ms')
    plt.legend()
    
    # 2. 상세 시간 분석
    if 'detailed_times' in results and results['detailed_times']:
        plt.subplot(2, 3, 2)
        components = ['backbone', 'attention', 'classification']
        means = [np.mean(results['detailed_times'][comp]) for comp in components]
        colors = ['lightcoral', 'lightgreen', 'lightsalmon']
        bars = plt.bar(components, means, color=colors)
        plt.title('Component-wise Inference Time')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        # 값 표시
        for bar, mean_val in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean_val:.2f}ms', ha='center', va='bottom')
    
    # 3. 실행별 추론 시간 변화
    plt.subplot(2, 3, 3)
    run_data = results['basic_times'][:min(100, len(results['basic_times']))]
    plt.plot(run_data, 'o-', markersize=3, linewidth=1, color='darkblue')
    plt.title('Inference Time Over Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    
    # 4. 메모리 사용량
    if results.get('memory_stats'):
        plt.subplot(2, 3, 4)
        memory_data = pd.DataFrame(results['memory_stats'])
        plt.plot(memory_data['batch'], memory_data['memory_peak_mb'], 'o-', color='purple')
        plt.title('GPU Memory Usage')
        plt.xlabel('Batch')
        plt.ylabel('Memory (MB)')
        plt.grid(True, alpha=0.3)
    
    # 5. 박스플롯
    plt.subplot(2, 3, 5)
    if 'detailed_times' in results and results['detailed_times']:
        data_to_plot = [results['detailed_times'][comp] for comp in components]
        box_plot = plt.boxplot(data_to_plot, labels=components, patch_artist=True)
        colors = ['lightcoral', 'lightgreen', 'lightsalmon']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title('Component Time Distribution')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
    
    # 6. 통계 요약 텍스트
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""Model Performance Summary
    
Model: ShuffleNetV2 ({results['model_config']['model_size']})
Parameters: {results['parameters_M']:.3f}M
Batch Size: {results['model_config']['batch_size']}
Precision: {results['model_config']['precision']}

Timing Statistics:
Mean: {results['mean_time']:.3f} ms
Std:  {results['std_time']:.3f} ms
Min:  {results['min_time']:.3f} ms
Max:  {results['max_time']:.3f} ms
FPS:  {results['fps']:.1f}

Accuracy:
Test Acc: {results['accuracy']:.4f}
Balanced: {results['balanced_accuracy']:.4f}"""
    
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', 
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_timing_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, args):
    """결과 저장 (학습 코드의 저장 구조와 유사)"""
    if not args.save_results:
        return
    
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    
    # 결과 디렉토리 생성
    results_dir = './FER_Models/ShuffleNet/inference_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    base_filename = os.path.join(results_dir, 
                                f'shufflenetv2_channel_attention_inference_{args.model_size}_iter{args.iteration}_{current_time}')
    
    # CSV 저장 (학습 코드와 유사한 형식)
    csv_data = {
        'Model_Type': f'ShuffleNetV2_ChannelAttention_{args.model_size}',
        'Iteration': args.iteration,
        'Batch_Size': args.batch_size,
        'Precision': args.precision,
        'Embedding_Size': args.embedding_size,
        'Model_Params_M': results['parameters_M'],
        'Checkpoint_Accuracy': results.get('checkpoint_accuracy', 0),
        'Test_Accuracy': results['accuracy'],
        'Balanced_Accuracy': results['balanced_accuracy'],
        'Avg_Loss': results['avg_loss'],
        'Mean_Time_ms': results['mean_time'],
        'Std_Time_ms': results['std_time'],
        'Min_Time_ms': results['min_time'],
        'Max_Time_ms': results['max_time'],
        'FPS': results['fps'],
        'Checkpoint_Path': args.checkpoint_path,
        'Timestamp': current_time
    }
    
    # 상세 시간 정보 추가
    if 'detailed_times' in results and results['detailed_times']:
        for component in ['backbone', 'attention', 'classification']:
            csv_data[f'{component.capitalize()}_Mean_ms'] = np.mean(results['detailed_times'][component])
            csv_data[f'{component.capitalize()}_Std_ms'] = np.std(results['detailed_times'][component])
    
    df = pd.DataFrame([csv_data])
    csv_path = f'{base_filename}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # JSON 저장 (전체 상세 결과)
    json_path = f'{base_filename}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to {json_path}")
    
    # 시각화 저장
    if args.create_visualizations:
        create_timing_visualizations(results, base_filename)
        if 'y_true' in results and 'y_pred' in results:
            create_confusion_matrix(results['y_true'], results['y_pred'], 
                                  {'accuracy': results['accuracy'], 
                                   'balanced_accuracy': results['balanced_accuracy']}, 
                                  base_filename)

def main():
    args = parse_args()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("SHUFFLENETV2 CHANNEL ATTENTION INFERENCE SPEED MEASUREMENT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\nModel Configuration:")
    print(f"  Model: ShuffleNetV2 + Channel Attention")
    print(f"  Model Size: {args.model_size}")
    print(f"  Embedding Size: {args.embedding_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Precision: {args.precision}")
    print(f"  Iteration: {args.iteration}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print("=" * 80)
    
    # 랜덤 시드 설정 (학습 코드와 동일)
    iteration_seed = args.iteration - 1
    control_random_seed(iteration_seed)
    
    # 데이터셋 로드 및 분할 (학습 코드와 완전히 동일)
    label_file = os.path.join(args.raf_path, 'EmoLabel/list_patition_label.txt')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"RAF-DB label file not found: {label_file}")
        
    df = pd.read_csv(label_file, sep=' ', header=None, names=['name', 'label'])
    file_names = df['name'].values
    labels = df['label'].values - 1

    print(f"Dataset loaded: {len(file_names)} samples")
    print(f"Classes: {np.unique(labels)}")

    # 데이터 분할 (학습 코드와 동일한 방식)
    ss = ShuffleSplit(n_splits=10, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_names, labels))
    
    if args.iteration > len(splits):
        print(f"Error: Iteration {args.iteration} exceeds available splits ({len(splits)})")
        return
    
    train_val_indices, test_indices = splits[iteration_seed]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=args.val_size, random_state=iteration_seed, 
        stratify=labels[train_val_indices]
    )
    
    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")
    
    # 데이터 전처리 (학습 코드의 validation transform과 동일)
    val_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 테스트 데이터셋 생성
    test_dataset = RafDataSet(args.raf_path, phase='test', 
                             indices=test_indices, transform=val_transforms)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )
    
    # 모델 생성 및 체크포인트 로드
    model, checkpoint_accuracy = load_model_and_checkpoint(args.checkpoint_path, args, device)
    model.eval()
    
    # 모델 파라미터 수 계산 (학습 코드와 동일한 방식)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print(f'Total Parameters: {model_parameters:.3f}M')
    print(f'Model Features: ShuffleNetV2 Backbone + Single Channel Attention Head')
    
    # 정확도 측정
    accuracy, balanced_acc, avg_loss, y_true, y_pred = measure_model_accuracy(
        model, test_loader, device, args
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Balanced Test Accuracy: {balanced_acc:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # 워밍업 실행 (학습 코드 구조와 유사)
    print(f"\nPerforming {args.warm_up} warm-up runs...")
    with torch.no_grad():
        for _ in range(args.warm_up):
            for imgs, _ in tqdm(test_loader, desc="Warm-up", leave=False):
                imgs = imgs.to(device)
                if args.precision == 'fp16':
                    imgs = imgs.half()
                _ = model(imgs)
    
    torch.cuda.empty_cache()
    
    # 기본 추론 시간 측정
    basic_times = measure_inference_time_basic(model, test_loader, device, args)
    
    # 상세 추론 시간 측정 (옵션)
    detailed_times = None
    if args.measure_detailed:
        detailed_times = measure_inference_time_detailed(model, test_loader, device, args)
    
    # 메모리 사용량 측정 (옵션)
    memory_stats = measure_memory_usage(model, test_loader, device, args)
    
    # 결과 통계 계산
    mean_time = np.mean(basic_times)
    std_time = np.std(basic_times)
    min_time = np.min(basic_times)
    max_time = np.max(basic_times)
    fps = 1000 / mean_time
    
    # 결과 출력 (학습 코드 스타일과 유사)
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    print("\n" + "="*80)
    print("INFERENCE SPEED RESULTS")
    print("="*80)
    print(f"[{current_time}] ShuffleNetV2 Channel Attention Results (Iteration {args.iteration})")
    print(f"Model Type: ShuffleNetV2 Backbone + Single Channel Attention Head")
    print(f"Model Size: {args.model_size}")
    print(f"Parameters: {model_parameters:.3f}M")
    print(f"Embedding Size: {args.embedding_size}")
    print(f"Precision: {args.precision}")
    print(f"Batch Size: {args.batch_size}")
    
    print(f"\nAccuracy Results:")
    if checkpoint_accuracy > 0:
        print(f"Checkpoint Accuracy: {checkpoint_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    print(f"\nInference Time Statistics (ms per batch):")
    print(f"Mean: {mean_time:.4f} ms")
    print(f"Std Dev: {std_time:.4f} ms")
    print(f"Min: {min_time:.4f} ms")
    print(f"Max: {max_time:.4f} ms")
    print(f"FPS: {fps:.2f}")
    
    if args.batch_size > 1:
        print(f"\nPer-image Statistics:")
        print(f"Mean: {mean_time/args.batch_size:.4f} ms per image")
        print(f"Throughput: {args.batch_size * fps:.2f} images/sec")
    
    # 상세 시간 분석 출력
    if detailed_times:
        print(f"\nComponent-wise Timing Analysis:")
        total_detailed = np.mean(detailed_times['total'])
        for component, times in detailed_times.items():
            if component != 'total':
                mean_comp = np.mean(times)
                std_comp = np.std(times)
                percentage = (mean_comp / total_detailed) * 100
                print(f"  {component.capitalize()}: {mean_comp:.4f} ± {std_comp:.4f} ms ({percentage:.1f}%)")
        print(f"  Total (detailed): {total_detailed:.4f} ms")
    
    # 메모리 사용량 출력
    if memory_stats:
        avg_memory = np.mean([stat['memory_peak_mb'] for stat in memory_stats])
        max_memory = np.max([stat['memory_peak_mb'] for stat in memory_stats])
        print(f"\nGPU Memory Usage:")
        print(f"Average Peak Memory: {avg_memory:.2f} MB")
        print(f"Maximum Peak Memory: {max_memory:.2f} MB")
    
    # 학습 코드 스타일의 분류 보고서
    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    report = classification_report(y_true, y_pred, target_names=emotion_labels, digits=4)
    print(f"\nClassification Report:")
    print(report)
    
    # 결과 저장을 위한 딕셔너리 구성
    results = {
        'model_config': {
            'model_size': args.model_size,
            'batch_size': args.batch_size,
            'precision': args.precision,
            'embedding_size': args.embedding_size,
            'iteration': args.iteration
        },
        'parameters_M': model_parameters,
        'checkpoint_accuracy': checkpoint_accuracy,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'avg_loss': avg_loss,
        'basic_times': basic_times,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'detailed_times': detailed_times,
        'memory_stats': memory_stats,
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=emotion_labels, output_dict=True),
        'create_visualizations': args.create_visualizations
    }
    
    # 결과 저장
    save_results(results, args)
    
    print("\n" + "="*80)
    print("MEASUREMENT COMPLETED")
    print("="*80)
    print("Model Features Summary:")
    print("1. Backbone: ShuffleNetV2 with Spatial Glance (Stage2 & Stage3)")
    print("2. Head: Single Channel Attention")
    print("3. Architecture: 512 channels → Channel Attention → FC")
    print("4. Optimized for facial expression recognition")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()