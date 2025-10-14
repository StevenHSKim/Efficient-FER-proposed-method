import torch
import os
import time
import random
import numpy as np
from datetime import datetime
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def create_experiment_folder(args):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if args.exp_name:
        exp_name = f"{args.exp_name}_{timestamp}"
    else:
        exp_name = f"{args.model}_{args.dataset}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    
    base_exp_dir = './experiments'
    experiment_dir = os.path.join(base_exp_dir, exp_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    for directory in [experiment_dir, checkpoints_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    return {'experiment_dir': experiment_dir, 'checkpoints_dir': checkpoints_dir, 'results_dir': results_dir}

def measure_model_complexity(model, input_size=(3, 112, 112)):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = 0
    if THOP_AVAILABLE:
        dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return {"params_M": params / 1_000_000, "flops_G": flops / 1_000_000_000 if THOP_AVAILABLE else -1}

def measure_inference_speed(model, input_size=(3, 112, 112)):
    device = next(model.parameters()).device
    model.eval()
    avg_latency_gpu = -1
    if device.type == 'cuda':
        dummy_input_gpu = torch.randn(1, *input_size, device=device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions=100; timings_gpu = np.zeros((repetitions,))
        for _ in range(10): _=model(dummy_input_gpu)
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record(); _ = model(dummy_input_gpu); ender.record()
                torch.cuda.synchronize(); timings_gpu[rep] = starter.elapsed_time(ender)
        avg_latency_gpu = np.sum(timings_gpu) / repetitions
    
    cpu_model = model.to('cpu'); dummy_input_cpu = torch.randn(1, *input_size)
    repetitions=100; timings_cpu = np.zeros((repetitions,))
    for _ in range(10): _ = cpu_model(dummy_input_cpu)
    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.time(); _ = cpu_model(dummy_input_cpu); end_time = time.time()
            timings_cpu[rep] = (end_time - start_time) * 1000
    avg_latency_cpu = np.sum(timings_cpu) / repetitions
    model.to(device)
    return {"latency_gpu_ms": avg_latency_gpu, "latency_cpu_ms": avg_latency_cpu}
        
def measure_backbone_complexity(backbone_model, input_size=(3, 224, 224)):
    # Calculate the number of parameters (only trainable parameters)
    params = sum(p.numel() for p in backbone_model.parameters() if p.requires_grad)
    
    flops = 0
    if THOP_AVAILABLE:
        # Create a dummy input tensor on the same device as the model
        device = next(backbone_model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)

        # Use thop.profile to calculate FLOPs
        flops, _ = profile(backbone_model, inputs=(dummy_input,), verbose=False)
    
    return {
        "params_M": params / 1_000_000,
        "flops_G": flops / 1_000_000_000 if THOP_AVAILABLE else -1,
    }