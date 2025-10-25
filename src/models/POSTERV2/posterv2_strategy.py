from src.models.base.model_strategy import ModelStrategy

class PosterV2Strategy(ModelStrategy):
    """단일 출력, 단일 loss"""
    
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def compute_loss(self, model_output, targets, **kwargs):
        return self.criterion(model_output, targets)
    
    def get_predictions(self, model_output):
        return model_output