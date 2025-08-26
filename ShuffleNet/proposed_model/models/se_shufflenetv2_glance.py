import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


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


class BetterShuffleNet_ChannelAttention(nn.Module):
    """Complete model: Better ShuffleNet backbone + Channel Attention head"""
    def __init__(self, input_size=224, n_class=7, model_size='1.0x', embedding_size=136, 
                 use_mdconv=True, custom_stage_repeats=None, use_residual=False):
        super(BetterShuffleNet_ChannelAttention, self).__init__()
        
        # Backbone
        self.backbone = BetterShuffleNet(input_size=input_size, n_class=n_class, 
                                       model_size=model_size, embedding_size=embedding_size,
                                       use_mdconv=use_mdconv, custom_stage_repeats=custom_stage_repeats,
                                       use_residual=use_residual)
        
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
        if self.training:
            return out
        else:
            return out, conv_features, attention_features


# 테스트 및 사용 예시
if __name__ == "__main__":
    print("Testing BetterShuffleNet with First Method SE Application:")
    
    # 기본 사용 (residual 없음)
    print("\n1. Testing without residual connections:")
    model_no_res = BetterShuffleNet_ChannelAttention(input_size=112, model_size='1.0x', n_class=7, 
                                                    use_mdconv=True, use_residual=False)
    
    # Residual connection 사용
    print("\n2. Testing with residual connections:")
    model_with_res = BetterShuffleNet_ChannelAttention(input_size=112, model_size='1.0x', n_class=7, 
                                                      use_mdconv=True, use_residual=True)
    
    # 테스트 데이터
    test_data = torch.rand(2, 3, 112, 112)
    
    # Training mode
    model_with_res.train()
    out_train = model_with_res(test_data)
    print(f"\nTraining output size: {out_train.size()}")
    
    # Evaluation mode  
    model_with_res.eval()
    out_eval = model_with_res(test_data)
    print(f"Evaluation output: {type(out_eval)}, lengths: {len(out_eval) if isinstance(out_eval, tuple) else 'single tensor'}")
    
    print("\n✅ All tests passed! First method SE application implemented.")
    print("\n📋 Key changes:")
    print("- SE applied only from stage 2 onwards (like ShuffleNetV2_Plus)")
    print("- SE applied at the end of branch operations")
    print("- Residual connections remain optional and compatible")
    print("- Last SE layer applied after final conv (following first method)")
    print("- SE reduction ratio follows first method (1/4 instead of 1/2)")