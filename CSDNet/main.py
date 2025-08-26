import os
import time
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit, train_test_split
import random
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from models.proposed_sg import ProposedNet

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/userHome/userhome1/kimhaesung/FER_Models/datasets/raf-basic', help='Raf-DB dataset path.')
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--model', type=str, default='ProposedNet', choices=['ProposedNet'], help='RapidNet model variant')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of emotion classes.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for repeated random sampling')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience.')
    parser.add_argument('--exp_name', type=str, default='', help='Custom experiment name (optional)')
    return parser.parse_args()

def create_experiment_folder(args):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if args.exp_name:
        exp_name = f"{args.exp_name}_{timestamp}"
    else:
        exp_name = f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    base_exp_dir = '/userHome/userhome1/kimhaesung/FER_Models/FER_Models/RapidNet/experiments'
    experiment_dir = os.path.join(base_exp_dir, exp_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    for directory in [experiment_dir, checkpoints_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    return {
        'experiment_dir': experiment_dir, 'checkpoints_dir': checkpoints_dir,
        'results_dir': results_dir, 'exp_name': exp_name, 'timestamp': timestamp
    }

# 모델 복잡도 및 추론 시간 측정 함수
def measure_model_complexity(model, input_size=(3, 112, 112)):
    """모델의 파라미터와 FLOPs 측정"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = 0
    if THOP_AVAILABLE:
        dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    
    return {
        "params_M": params / 1_000_000,
        "flops_G": flops / 1_000_000_000 if THOP_AVAILABLE else -1,
    }

def measure_inference_speed(model, input_size=(3, 112, 112)):
    """학습된 모델의 추론 시간을 측정"""
    device = next(model.parameters()).device
    model.eval()

    # GPU 추론 시간 측정
    avg_latency_gpu = -1
    if device.type == 'cuda':
        dummy_input_gpu = torch.randn(1, *input_size, device=device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings_gpu = np.zeros((repetitions,))
        for _ in range(10): _ = model(dummy_input_gpu) # 워밍업
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input_gpu)
                ender.record()
                torch.cuda.synchronize()
                timings_gpu[rep] = starter.elapsed_time(ender)
        avg_latency_gpu = np.sum(timings_gpu) / repetitions

    # CPU 추론 시간 측정
    cpu_model = model.to('cpu')
    dummy_input_cpu = torch.randn(1, *input_size)
    repetitions = 100
    timings_cpu = np.zeros((repetitions,))
    for _ in range(10): _ = cpu_model(dummy_input_cpu) # 워밍업
    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.time()
            _ = cpu_model(dummy_input_cpu)
            end_time = time.time()
            timings_cpu[rep] = (end_time - start_time) * 1000 # ms
    avg_latency_cpu = np.sum(timings_cpu) / repetitions
    
    model.to(device) # 모델을 원래 디바이스로 복원
    return {
        "latency_gpu_ms": avg_latency_gpu,
        "latency_cpu_ms": avg_latency_cpu,
    }

def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, indices, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        label_file = os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt')
        df = pd.read_csv(label_file, sep=' ', header=None, names=['name', 'label'])
        self.file_names = df['name'].values[indices]
        self.labels = df['label'].values[indices] - 1
        self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in self.file_names]
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def train(args, train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, patience, iteration, checkpoint_dir):
    best_loss = float('inf')
    best_acc = 0
    patience_counter = 0
    for epoch in tqdm(range(1, epochs + 1), desc=f"Training Iteration {iteration+1}"):
        model.train()
        for (imgs, targets) in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        if scheduler: scheduler.step()
        val_loss, acc, balanced_acc = validate(val_loader, model, criterion, device)
        tqdm.write(f"[Epoch {epoch}] Val Acc: {acc:.4f}, bAcc: {balanced_acc:.4f}, Loss: {val_loss:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_iter_{iteration+1}.pth"))
            tqdm.write(f"New best model saved with acc: {best_acc:.4f}")
        if best_loss > val_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            tqdm.write("Early stopping triggered")
            break
    return os.path.join(checkpoint_dir, f"best_model_iter_{iteration+1}.pth"), best_acc

def validate(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for (imgs, targets) in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            loss = criterion(out, targets)
            val_loss += loss.item()
            _, predicts = torch.max(out, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())
    val_loss /= len(val_loader)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return val_loss, acc, balanced_acc

def test(test_loader, model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for (imgs, targets) in tqdm(test_loader, desc="Testing"):
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            _, predicts = torch.max(out, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, balanced_acc, cm

def run_train_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment_paths = create_experiment_folder(args)
    
    print(f"Experiment: {experiment_paths['exp_name']}")
    print(f"Device: {device}")
    if not THOP_AVAILABLE:
        print("Warning: `thop` is not installed. FLOPs calculation will be skipped. Install with `pip install thop`")

    df = pd.read_csv(os.path.join(args.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    
    all_results = []
    model_complexity = {}
    final_checkpoint_path = None # 마지막 체크포인트 경로 저장 변수
    
    model_factory = {'ProposedNet': ProposedNet}
    if args.model not in model_factory:
        raise ValueError(f"Model {args.model} not supported!")

    # --- 모델 복잡도 측정 (실험 시작 시 1회) ---
    print(f"Creating model: {args.model} for complexity measurement.")
    temp_model = model_factory[args.model](num_classes=args.num_classes).to(device)
    model_complexity = measure_model_complexity(temp_model, input_size=(3, 112, 112))
    print("="*50)
    print("Model Complexity Report")
    print(f"  - Parameters: {model_complexity['params_M']:.2f}M")
    if THOP_AVAILABLE: print(f"  - FLOPs: {model_complexity['flops_G']:.2f}G")
    print("="*50)
    del temp_model # 메모리 확보

    for iteration, (train_val_indices, test_indices) in enumerate(ss.split(df)):
        print(f"\n{'='*25} Iteration {iteration + 1}/{args.iterations} {'='*25}")
        control_random_seed(iteration)

        model = model_factory[args.model](num_classes=args.num_classes).to(device)
        
        # 데이터셋 및 로더 설정
        data_transforms = transforms.Compose([
            transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25))])
        val_transforms = transforms.Compose([
            transforms.Resize((112, 112)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration)
        
        train_dataset = RafDataSet(args.raf_path, 'train', train_indices, data_transforms)
        val_dataset = RafDataSet(args.raf_path, 'validation', val_indices, val_transforms)
        test_dataset = RafDataSet(args.raf_path, 'test', test_indices, val_transforms)
        
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_checkpoint_path, _ = train(
            args, train_loader, val_loader, model, criterion, optimizer, scheduler,
            device, args.epochs, args.early_stopping_patience, iteration, experiment_paths['checkpoints_dir']
        )
        
        if os.path.exists(best_checkpoint_path):
            test_acc, test_balanced_acc, _ = test(test_loader, model, best_checkpoint_path, device)
            all_results.append({'iter': iteration + 1, 'acc': test_acc, 'bacc': test_balanced_acc})
            final_checkpoint_path = best_checkpoint_path # 마지막 성공한 체크포인트 경로 업데이트
            print(f"Iteration {iteration+1} Test Acc: {test_acc:.4f}, Test Balanced Acc: {test_balanced_acc:.4f}")
        else:
            print(f"No valid checkpoint found for iteration {iteration + 1}")
    
    # --- 추론 시간 측정 (모든 실험 종료 후 1회) ---
    inference_speed = {}
    if final_checkpoint_path and os.path.exists(final_checkpoint_path):
        print("\n" + "="*80)
        print(f"Measuring inference speed on final best model: {os.path.basename(final_checkpoint_path)}")
        final_model = model_factory[args.model](num_classes=args.num_classes)
        final_model.load_state_dict(torch.load(final_checkpoint_path, map_location=device))
        final_model.to(device)
        inference_speed = measure_inference_speed(final_model)
        print(f"  - GPU Latency: {inference_speed['latency_gpu_ms']:.2f} ms")
        print(f"  - CPU Latency: {inference_speed['latency_cpu_ms']:.2f} ms")
    else:
        print("\nCould not measure inference speed: No final model checkpoint found.")

    # --- 최종 결과 저장 및 출력 ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary = {
            "Model": args.model,
            "Mean Accuracy": f"{results_df['acc'].mean():.4f}", "Std Accuracy": f"{results_df['acc'].std():.4f}",
            "Mean Balanced Accuracy": f"{results_df['bacc'].mean():.4f}", "Std Balanced Accuracy": f"{results_df['bacc'].std():.4f}",
            "Parameters (M)": f"{model_complexity.get('params_M', -1):.2f}",
            "FLOPs (G)": f"{model_complexity.get('flops_G', -1):.2f}",
            "GPU Latency (ms)": f"{inference_speed.get('latency_gpu_ms', -1):.2f}",
            "CPU Latency (ms)": f"{inference_speed.get('latency_cpu_ms', -1):.2f}",
            **vars(args)}
        
        results_df.to_csv(os.path.join(experiment_paths['results_dir'], 'iteration_results.csv'), index=False)
        with open(os.path.join(experiment_paths['results_dir'], 'summary.txt'), 'w') as f:
            for key, value in summary.items(): f.write(f"{key}: {value}\n")
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        for key, value in summary.items():
            if key in ["Model", "Mean Accuracy", "Std Accuracy", "Parameters (M)", "FLOPs (G)", "GPU Latency (ms)", "CPU Latency (ms)"]:
                print(f"- {key}: {value}")
        print("="*80)
        print(f"All results saved in: {experiment_paths['experiment_dir']}")

if __name__ == "__main__":
    run_train_test()