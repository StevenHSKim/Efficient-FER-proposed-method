import torch.nn as nn
import sys
import os

# --- 1. 표준 ResNet 임포트 (폴백용) ---
# GResNet.py의 resnet18과 이름 충돌을 피하기 위해 별칭(alias) 사용
from torchvision.models import resnet18 as standard_resnet18, resnet34 as standard_resnet34

# --- 2. CSE-GResNet 모듈 임포트 ---
# 이 파일(model.py)은 GResNet.py, Gabor_CNN_PyTorch 등과
# 같은 'models' 패키지 안에 있습니다.
try:
    # 2-1. models/GResNet.py 에서 임포트
    # (같은 폴더에 있으므로 상대 경로 .GResNet 사용)
    from .GResNet import resnet18, resnet34
    
    # 2-2. models/Channel_Shift_Enhance_Module.py 에서 임포트
    from .Channel_Shift_Enhance_Module import make_channel_shift
    
    # 2-3. models/Gabor_CNN_PyTorch/gcn/layers/GConv.py 에서 임포트
    from .Gabor_CNN_PyTorch.gcn.layers.GConv import GConv 
    
    GABOR_CNN_AVAILABLE = True
    print("Successfully imported CSE-GResNet modules (GResNet, Channel_Shift, GConv).")
    
except ImportError as e:
    print(f"Warning: CSE-GResNet modules import failed ({e}). Using standard ResNet fallback.")
    GABOR_CNN_AVAILABLE = False
    
    # GABOR_CNN_AVAILABLE이 False일 때 사용할 더미 함수 정의
    def resnet18(pretrained=False, **kwargs):
        raise ImportError("GResNet.py not found or failed to import")
    def resnet34(pretrained=False, **kwargs):
        raise ImportError("GResNet.py not found or failed to import")
    def make_channel_shift(model, args=None):
        print("Channel shift module not available. Using original model.")
        return model

# --- 3. 내부 ARGS 클래스 (원본 main.py에서 이동) ---
# 이 모델의 고정된 내부 설정을 정의합니다.
class Args:
    def __init__(self):
        self.model = 'GCN'
        self.GCN_is_maxpool = False
        self.base_model = 'resnet18'
        self.TSM_position = ['layer3']
        self.TSM_div = 8
        self.TSM_module_insert = 'residual'
        self.TSM_channel_enhance = 4
        self.channel_block_position = 'stochastic'
        self.TSM_div_e = 16
        self.channel_block_position_e = 'stochastic'
        self.channel_enhance_init_strategy = 3
        self.channel_enhance_kernelsize = 3
        self.TSM_conv_insert_e = 'residual'
        self.fusion_SE = 'A'
        self.GConv_M = 4

# --- 4. 폴백(Fallback)용 표준 ResNet 생성 함수 ---
def create_standard_resnet_model(num_classes, base_model='resnet18'):
    """표준 torchvision ResNet 모델을 생성합니다."""
    if base_model == 'resnet18': 
        model = standard_resnet18(pretrained=False, num_classes=num_classes)
    elif base_model == 'resnet34': 
        model = standard_resnet34(pretrained=False, num_classes=num_classes)
    else: 
        raise ValueError(f"Unsupported base model: {base_model}")
    return model

# --- 5. 팩토리(Factory) 함수 (main.py에서 호출할 함수) ---
def CSE_GResNet(num_classes, **kwargs):
    """
    (공장 함수) CSE-GResNet 모델을 생성합니다.
    실패 시 표준 ResNet으로 대체(fallback)하며,
    모델 객체와 로깅용 모델 이름을 반환합니다.
    """
    
    # 1. 모델의 내부 설정값(Args) 인스턴스 생성
    model_args = Args() 
    
    model = None
    model_name_log = "" # 로깅에 사용될 모델 이름
    
    # 2. CSE-GResNet 생성 시도
    try:
        if GABOR_CNN_AVAILABLE:
            # GResNet.py의 resnet18/34 함수에 전달할 인자 준비
            model_kwargs = {
                'num_classes': num_classes, 
                'M': model_args.GConv_M, 
                'nScale': 3,
                'GCN_is_maxpool': model_args.GCN_is_maxpool, 
                'input_channel': 3,
                'args': model_args  # GResNet.py의 함수들이 필요로 하는 args 객체
            }
            
            base_name = model_args.base_model.replace('resnet', '')
            
            # GResNet.py에서 임포트한 resnet18 또는 resnet34 호출
            if model_args.base_model == 'resnet18':
                model = resnet18(**model_kwargs)
            else:
                model = resnet34(**model_kwargs)
            
            # Channel_Shift_Enhance_Model.py에서 임포트한 함수 적용
            model = make_channel_shift(model, model_args) 
            
            model_name_log = f"CSE-GResNet-{base_name}"
        else:
            # Gabor 모듈 로드에 실패했으면 예외 발생
            raise ImportError("Gabor CNN not available (trigger fallback)")
            
    # 3. 실패 시 폴백 로직
    except Exception as e:
        print(f"Error during Gabor model creation ({e}). Using standard ResNet fallback.")
        base_name = model_args.base_model.replace('resnet', '')
        model_name_log = f"Standard-ResNet-{base_name}"
        model = create_standard_resnet_model(num_classes, model_args.base_model)
    
    print(f"Using {model_name_log} model")
    
    # 4. (모델, 모델 이름) 튜플 반환
    return model, model_name_log