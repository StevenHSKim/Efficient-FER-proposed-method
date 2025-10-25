import torch
import torch.nn.functional as F
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def generate_adaptive_LD(outputs, targets, num_classes, threshold, sharpen, T):
    device = outputs.device
    outputs_softmax = F.softmax(outputs, dim=1).detach()
    LD = torch.zeros(num_classes, num_classes).to(device)

    for i in range(num_classes):
        if (targets == i).any():
            LD[i] = outputs_softmax[targets == i].mean(dim=0)
        else:
            # Handle classes with no samples
            LD[i] = torch.zeros(num_classes).to(device).fill_((1 - threshold) / (num_classes - 1)).scatter_(0, torch.tensor(i).to(device), threshold)

    if sharpen:
        LD = torch.pow(LD, 1 / T) / torch.sum(torch.pow(LD, 1 / T), dim=1, keepdim=True)

    return LD

def generate_average_weights(weights, targets, num_classes, max_weight, min_weight):
    device = weights.device
    weights_avg = torch.zeros(num_classes).to(device)
    weights_max = torch.zeros(num_classes).to(device)
    weights_min = torch.zeros(num_classes).to(device)
    
    nan = float('nan')
    weights_avg_list = [nan for _ in range(num_classes)]
    weights_max_list = [nan for _ in range(num_classes)]
    weights_min_list = [nan for _ in range(num_classes)]

    for i in range(num_classes):
        if (targets == i).any():
            weights_avg[i] = weights[targets == i].mean()
            weights_max[i] = weights[targets == i].max()
            weights_min[i] = weights[targets == i].min()
            
            weights_avg_list[i] = weights_avg[i].item()
            weights_max_list[i] = weights_max[i].item()
            weights_min_list[i] = weights_min[i].item()

    return weights_avg_list, weights_max_list, weights_min_list