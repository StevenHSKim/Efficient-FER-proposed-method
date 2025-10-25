import copy

import timm
import torch
import torch.nn as nn

def create_model(num_classes=7, drop_rate=0):
    model = ResNet18(num_classes=num_classes, drop_rate=drop_rate)

    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate

        model = timm.create_model('resnet18', pretrained=False)
        # 사전학습 가중치 안 쓰고 실험
        # checkpoint = torch.load('/userHome/userhome1/kimhaesung/FER_Models/FER_Models/DAN/models/resnet18_msceleb.pth')
        # model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.fc = nn.Linear(512, num_classes)

        self.feature = nn.Sequential(*list(model.children())[:-5])
        
        # -------------- 백본 가중치 동결 여부  -------------- #
        # for param in self.feature.parameters():
        #     param.requires_grad = False
        # ------------------------------------------------- #

        self.branch_1 = nn.Sequential(*list(model.children())[-5:])
        self.branch_1_feature = nn.Sequential(*list(self.branch_1.children())[:-1])
        self.branch_1_classifier = self.branch_1[-1]
        
        self.branch_2 = copy.deepcopy(self.branch_1)
        self.branch_2_feature = nn.Sequential(*list(self.branch_2.children())[:-1])
        self.branch_2_classifier = self.branch_2[-1]
        
        self.alpha_1 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())    
        self.alpha_2 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())  

    def forward(self, image):
        feature_0 = self.feature(image)
        feature_0 = nn.Dropout(self.drop_rate)(feature_0)

        feature_1 = self.branch_1_feature(feature_0)
        feature_2 = self.branch_2_feature(feature_0)

        attention_weights_1 = self.alpha_1(feature_1)
        attention_weights_2 = self.alpha_2(feature_2)

        out_1 = attention_weights_1 * self.branch_1_classifier(feature_1)
        out_2 = attention_weights_2 * self.branch_2_classifier(feature_2)

        attention_weights = (attention_weights_1 + attention_weights_2) / 2
        
        return out_1, out_2, attention_weights
    
if __name__ == "__main__":
    print("Testing ResNet18 with Dual Branch...")
    model = create_model(num_classes=7, drop_rate=0.5)

    # 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_data = torch.rand(1, 3, 224, 224, device=device)

    from thop import profile
    macs, params_thop = profile(model, inputs=(test_data,), verbose=False)
    flops = macs * 2  # 1 MAC = 2 FLOPs
    print(f"MACs:  {macs/1e6:.2f} M")
    print(f"FLOPs: {flops/1e9:.2f} G")