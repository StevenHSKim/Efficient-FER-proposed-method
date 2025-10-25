from src.models.base.model_strategy import ModelStrategy

class DANStrategy(ModelStrategy):
    """DAN용 (3개 출력, 3개 loss)"""
    
    def __init__(self, criterion_cls, criterion_af, criterion_pt):
        super().__init__()
        self.criterion_cls = criterion_cls
        self.criterion_af = criterion_af
        self.criterion_pt = criterion_pt
    
    def compute_loss(self, model_output, targets, **kwargs):
        out, feat, heads = model_output
        loss = (self.criterion_cls(out, targets) + 
                self.criterion_af(feat, targets) + 
                self.criterion_pt(heads))
        return loss
    
    def get_predictions(self, model_output):
        out, _, _ = model_output
        return out