import torch
import torch.nn.functional as F
import math
from src.models.base.model_strategy import ModelStrategy
from src.models.AdaDF.adadf_utils import generate_adaptive_LD

class AdaDFStrategy(ModelStrategy):
    """Ada-DF용 (LD 업데이트 필요)"""
    
    def __init__(self, criterion, criterion_kld, args, device):
        super().__init__()
        self.criterion = criterion
        self.criterion_kld = criterion_kld
        self.args = args
        self.device = device
        self.LD = None # Stateful
        
    def initialize_LD(self, num_classes):
        """LD 초기화"""
        LD = torch.zeros(num_classes, num_classes).to(self.device)
        for i in range(num_classes):
            LD[i] = torch.zeros(num_classes).fill_(
                (1 - self.args.threshold) / (num_classes - 1)
            ).scatter_(0, torch.tensor(i), self.args.threshold)
        if self.args.sharpen:
            LD = torch.pow(LD, 1 / self.args.T) / torch.sum(
                torch.pow(LD, 1 / self.args.T), dim=1, keepdim=True # keepdim 추가
            )
        self.LD = LD
    
    def compute_loss(self, model_output, targets, **kwargs):
        outputs_1, outputs_2, attention_weights = model_output
        batch_size = targets.size(0)
        
        # Dynamic alpha
        if self.args.alpha is not None:
            alpha_1, alpha_2 = self.args.alpha, 1 - self.args.alpha
        else:
            if self.epoch <= self.args.beta:
                alpha_1 = math.exp(-(1 - self.epoch / self.args.beta) ** 2)
                alpha_2 = 1
            else:
                alpha_1 = 1
                alpha_2 = math.exp(-(1 - self.args.beta / self.epoch) ** 2)
        
        # RR Loss
        tops = int(batch_size * self.args.tops)
        _, top_idx = torch.topk(attention_weights.squeeze(), tops)
        _, down_idx = torch.topk(attention_weights.squeeze(), 
                                 batch_size - tops, largest=False)
        high_group = attention_weights[top_idx]
        low_group = attention_weights[down_idx]
        diff = low_group.mean() - high_group.mean() + self.args.margin_1
        RR_loss = diff if diff > 0 else torch.tensor(0.0).to(self.device)
        
        # CE Loss
        loss_ce = self.criterion(outputs_1, targets).mean()
        
        # Attention weights normalization
        attention_weights = attention_weights.squeeze(1)
        attention_weights = (
            (attention_weights - attention_weights.min()) /
            (attention_weights.max() - attention_weights.min() + eps)
        ) * (self.args.max_weight - self.args.min_weight) + self.args.min_weight
        attention_weights = attention_weights.unsqueeze(1)
        
        # KLD Loss
        self.LD = self.LD.to(targets.device) # DDP/DataParallel 호환성
        targets_dist = (
            (1 - attention_weights) * F.softmax(outputs_1, dim=1) +
            attention_weights * self.LD[targets]
        )
        loss_kld = self.criterion_kld(
            F.log_softmax(outputs_2, dim=1), targets_dist
        ).sum() / batch_size
        
        total_loss = alpha_2 * loss_ce + alpha_1 * loss_kld + RR_loss
        
        # 추가 정보 저장 (LD 업데이트용)
        return total_loss, {
            'outputs_1': outputs_1.detach(),
            'targets': targets.detach(),
            'attention_weights': attention_weights.detach()
        }
    
    def get_predictions(self, model_output):
        _, out, _ = model_output # Validation/Test에서는 out2 사용
        return out
    
    def update_LD(self, all_outputs, all_targets, num_classes):
        """에폭 종료 후 LD 업데이트"""
        self.LD = generate_adaptive_LD(
            all_outputs, all_targets, num_classes,
            self.args.threshold, self.args.sharpen, self.args.T
        )
