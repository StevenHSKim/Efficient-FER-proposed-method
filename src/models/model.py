import torch
import torch.nn as nn
import torch.nn.functional as F

class hard_swish(nn.Module):
    def forward(self, x): return x * F.relu6(x + 3, inplace=True) / 6

class hard_sigmoid(nn.Module):
    def forward(self, x): return F.relu6(x + 3, inplace=True) / 6

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False), hard_sigmoid()
        )
    def forward(self, x): 
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class Stem(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(i, o//2, 3, 2, 1), nn.BatchNorm2d(o//2), hard_swish(), nn.Conv2d(o//2, o, 3, 1, 1), nn.BatchNorm2d(o), hard_swish())
    def forward(self, x):
        return self.stem(x)

class Downsample(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o))
    def forward(self, x): return self.conv(x)

class GatedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(GatedSpatialAttention, self).__init__()
        intermediate_channels = in_channels // reduction

        self.global_context_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, intermediate_channels, 1, bias=False),
            hard_swish(),
            nn.Conv2d(intermediate_channels, intermediate_channels, 1, bias=False),
            hard_sigmoid()
        )
        
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.act1 = hard_swish()
        
        self.conv2 = nn.Conv2d(intermediate_channels, 1, 7, 1, 3, bias=False)
        self.act2 = hard_sigmoid()

    def forward(self, x):
        mod_vector = self.global_context_modulator(x)
        x_inter = self.conv1(x)
        x_inter = self.bn1(x_inter)
        x_mod = x_inter * mod_vector
        x_att = self.act1(x_mod)
        attention_map = self.conv2(x_att)
        attention_map = self.act2(attention_map)
        return x * attention_map

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, expansion_ratio=4):
        super().__init__(); expanded_dim=int(in_dim*expansion_ratio)
        self.pw1=nn.Conv2d(in_dim, expanded_dim, 1, bias=False)
        self.norm1=nn.BatchNorm2d(expanded_dim)
        self.act1=hard_swish()
        self.dw=nn.Conv2d(expanded_dim, expanded_dim, kernel, 1, 1, groups=expanded_dim, bias=False)
        self.norm2=nn.BatchNorm2d(expanded_dim)
        self.act2=hard_swish()
        self.pw2=nn.Conv2d(expanded_dim, out_dim, 1, bias=False)
        self.norm3=nn.BatchNorm2d(out_dim)
    def forward(self, x):
        return self.norm3(self.pw2(self.act2(self.norm2(self.dw(self.act1(self.norm1(self.pw1(x))))))))

class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., use_se=False):
        super().__init__(); self.use_se=use_se
        self.dws=DepthWiseSeparable(in_dim=dim, out_dim=dim, kernel=kernel, expansion_ratio=expansion_ratio)
        if self.use_se: self.se = SELayer(dim)
    def forward(self, x):
        out = self.dws(x)
        if self.use_se: out=self.se(out)
        return x + out

class MDConv(nn.Module):
    def __init__(self, channels, stride=1):
        super(MDConv, self).__init__()
        base_channels = channels // 3
        remainder = channels % 3
        self.group_channels = [base_channels + 1 if i < remainder else base_channels for i in range(3)]
        
        self.conv3x3_d1 = nn.Conv2d(self.group_channels[0], self.group_channels[0], 3, stride, 1, 1, groups=self.group_channels[0], bias=False)
        self.conv3x3_d2 = nn.Conv2d(self.group_channels[1], self.group_channels[1], 3, stride, 2, 2, groups=self.group_channels[1], bias=False)
        self.conv3x3_d3 = nn.Conv2d(self.group_channels[2], self.group_channels[2], 3, stride, 3, 3, groups=self.group_channels[2], bias=False)
        
        self.norm = nn.BatchNorm2d(channels)
        self.act = hard_swish()

    def forward(self, x):
        x_splits = torch.split(x, self.group_channels, dim=1)
        out = torch.cat([
            self.conv3x3_d1(x_splits[0]),
            self.conv3x3_d2(x_splits[1]),
            self.conv3x3_d3(x_splits[2])
        ], dim=1)
        return self.act(self.norm(out))

class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvFFN, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=in_channels, bias=False))
        self.norm1=nn.BatchNorm2d(out_channels)
        self.fc1=nn.Conv2d(in_channels, hidden_channels, 1)
        self.norm2=nn.BatchNorm2d(hidden_channels)
        self.act=hard_swish()
        self.fc2=nn.Conv2d(hidden_channels, out_channels, 1)
        self.norm3=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.norm3(self.fc2(self.act(self.norm2(self.fc1(self.norm1(self.conv(x)))))))

class DM_Block(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mixer = MDConv(channels=in_dim)
        self.ffn = ConvFFN(in_dim, in_dim * 4, in_dim)
    def forward(self, x):
        out = self.mixer(x)
        out = self.ffn(out)
        return x + out

class ProposedNet_Backbone(torch.nn.Module):
    def __init__(self, blocks, channels, emb_dims=512):
        super(ProposedNet_Backbone, self).__init__()
        self.stem = Stem(3, channels[0])
        
        self.stages = nn.ModuleList()
        in_channels = channels[0]
        for i in range(len(blocks)):
            stage_blocks = []
            local_stages, global_stages = blocks[i]
            out_channels = channels[i]
            if i > 0:
                stage_blocks.append(Downsample(in_channels, out_channels))
            use_se_in_stage = (i >= 1)
            for _ in range(local_stages):
                stage_blocks.append(InvertedResidual(out_channels, 3, 4., use_se=use_se_in_stage))
            for _ in range(global_stages):
                stage_blocks.append(DM_Block(out_channels))
            self.stages.append(nn.Sequential(*stage_blocks))
            in_channels = out_channels
        
        self.spatial_glance = GatedSpatialAttention(channels[1])
        self.conv_last = nn.Sequential(nn.Conv2d(channels[-1], emb_dims, 1, bias=True), nn.BatchNorm2d(emb_dims), hard_swish())
        
        # 7x7 to 1x1
        self.gdc_layer = nn.Sequential(
            nn.Conv2d(emb_dims, emb_dims, kernel_size=7, groups=emb_dims, bias=False),
            nn.BatchNorm2d(emb_dims)
        )
        
        self.model_init()
        
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.spatial_glance(x)
        x = self.stages[2](x)
        x = self.conv_last(x)  # output: [B, emb_dims, 7, 7]
        x = self.gdc_layer(x)  # output: [B, emb_dims, 1, 1]
        return x

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super().__init__()
        self.attention=nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(input_channels, input_channels//reduction, 1, bias=False), 
            nn.ReLU(True), 
            nn.Conv2d(input_channels//reduction, input_channels, 1, bias=False), 
            hard_sigmoid()
        )
    def forward(self, x):
        return x * self.attention(x)

class ProposedFERNet(nn.Module):
    def __init__(self, blocks, channels, emb_dims=512, num_classes=7):
        super(ProposedFERNet, self).__init__()
        self.blocks = blocks
        self.backbone = ProposedNet_Backbone(blocks, channels, emb_dims)
        
        self.head = nn.Sequential(
            ChannelAttention(emb_dims),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(emb_dims, num_classes)
        )
            
    def forward(self, x):
        features = self.backbone(x)  # output: [B, C, 1, 1]
        logits = self.head(features)
        return logits
    
def ProposedNet(num_classes=7, **kwargs):
    return ProposedFERNet(blocks=[[3,0], [2,4], [2,2]], 
                          channels=[48, 96, 160], 
                          emb_dims=512, 
                          num_classes=num_classes)