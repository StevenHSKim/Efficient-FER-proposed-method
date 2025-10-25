import torch
import torch.nn as nn

class ModelStrategy:
    """모델별 특수 로직을 캡슐화하는 베이스 클래스"""
    
    def __init__(self):
        self.epoch = 0 # AdaDF에서 사용
        
    def compute_loss(self, model_output, targets, **kwargs):
        """Loss 계산 (모델마다 다름)"""
        raise NotImplementedError
    
    def get_predictions(self, model_output):
        """예측값 추출 (모델마다 출력 구조가 다름)"""
        raise NotImplementedError
    
    def forward_model(self, model, images):
        """모델 forward (기본 구현)"""
        return model(images)
    
def get_model_strategy(model_name, args, device, num_classes):
    """모델별 전략 객체 생성"""
    from src.models.DAN.dan_strategy import DANStrategy
    from src.models.AdaDF.adadf_strategy import AdaDFStrategy
    from src.models.ProposedNet.proposednet_strategy import ProposedNetStrategy
    from src.models.POSTERV2.posterv2_strategy import PosterV2Strategy

    from src.models.DAN.dan_losses import AffinityLoss, PartitionLoss
            
    if model_name == 'DAN':
        criterion_cls = torch.nn.CrossEntropyLoss()
        # AffinityLoss는 state(파라미터)를 가지므로 device 전달이 중요
        criterion_af = AffinityLoss(device, num_class=num_classes)
        criterion_pt = PartitionLoss()
        return DANStrategy(criterion_cls, criterion_af, criterion_pt)
    
    elif model_name == 'AdaDF':
        criterion = nn.CrossEntropyLoss(
            reduction='none', 
            label_smoothing=args.label_smoothing
        )
        criterion_kld = nn.KLDivLoss(reduction='none')
        strategy = AdaDFStrategy(criterion, criterion_kld, args, device)
        strategy.initialize_LD(num_classes)
        return strategy
    
    elif model_name == 'ProposedNet':
        criterion = torch.nn.CrossEntropyLoss()
        return ProposedNetStrategy(criterion)
    
    elif model_name == 'PosterV2':
        criterion = torch.nn.CrossEntropyLoss()
        return PosterV2Strategy(criterion)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")