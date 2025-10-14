import os
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
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
import matplotlib
matplotlib.use('Agg')

from models.model import ProposedNet
from dataset import load_dataset_info, create_datasets
from utils import create_experiment_folder, measure_model_complexity, measure_inference_speed, control_random_seed, THOP_AVAILABLE

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Facial Emotion Recognition Training Script")
    
    # dataset arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['ckplus', 'expw', 'fer2013', 'ferplus', 'rafdb', 'sfew'], help='Select the dataset to use.')
    parser.add_argument('--data_path', type=str, default='../datasets', help='Path to the dataset directory or the main CSV file.')

    # common arguments
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--model', type=str, default='ProposedNet', choices=['ProposedNet'], help='Model variant.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of training data to use for validation')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for repeated random sampling.')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience.')
    parser.add_argument('--exp_name', type=str, default='', help='Custom experiment name (optional)')
    return parser.parse_args()


# Training, Validation, Testing Functions
def train(args, train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, patience, iteration, checkpoint_dir):
    best_loss, best_acc, patience_counter = float('inf'), 0, 0
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
            best_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            tqdm.write("Early stopping triggered")
            break
    return os.path.join(checkpoint_dir, f"best_model_iter_{iteration+1}.pth"), best_acc

def validate(val_loader, model, criterion, device):
    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []
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

# Main Execution Function
def run_train_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment_paths = create_experiment_folder(args)
    
    print(f"Starting Experiment: {os.path.basename(experiment_paths['experiment_dir'])}")
    print(f" - Dataset: {args.dataset}")
    print(f" - Device: {device}")
    if not THOP_AVAILABLE:
        print("Warning: `thop` is not installed. FLOPs calculation will be skipped.")
    
    # Dataset loading and preprocessing
    try:
        all_data_indices, all_labels, num_classes, use_stratify = load_dataset_info(args)
        print(f"Successfully loaded {args.dataset} data. Total samples: {len(all_labels)}. Auto-detected {num_classes} classes.")
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        return

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    all_results = []
    
    # Calculate and display model complexity once
    temp_model = ProposedNet(num_classes=num_classes).to(device)
    model_complexity = measure_model_complexity(temp_model)
    model_blocks_cfg = getattr(temp_model, 'blocks', 'N/A')
    print("="*50)
    print("Model Complexity Report")
    print(f"  - Parameters: {model_complexity['params_M']:.2f}M")
    if THOP_AVAILABLE: print(f"  - FLOPs: {model_complexity['flops_G']:.2f}G")
    print(f"  - Blocks Cfg: {model_blocks_cfg}")
    print("="*50)
    del temp_model

    # Iterative training and testing
    for iteration, (train_val_indices, test_indices) in enumerate(ss.split(all_data_indices, all_labels)):
        print(f"\n{'='*25} Iteration {iteration + 1}/{args.iterations} {'='*25}")
        control_random_seed(iteration)

        model = ProposedNet(num_classes=num_classes).to(device)
        
        data_transforms = transforms.Compose([
            transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25))])
        val_transforms = transforms.Compose([
            transforms.Resize((112, 112)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # Split methods of each dataset
        train_dataset, val_dataset, test_dataset = create_datasets(args, train_val_indices, test_indices, all_data_indices, all_labels, use_stratify, iteration, data_transforms, val_transforms)
        
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_checkpoint_path, _ = train(args, train_loader, val_loader, model, criterion, optimizer, scheduler, device, args.epochs, args.early_stopping_patience, iteration, experiment_paths['checkpoints_dir'])
        
        if os.path.exists(best_checkpoint_path):
            test_acc, test_balanced_acc, _ = test(test_loader, model, best_checkpoint_path, device)
            all_results.append({'iter': iteration + 1, 'acc': test_acc, 'bacc': test_balanced_acc})
            print(f"Iteration {iteration+1} Test Acc: {test_acc:.4f}, Test Balanced Acc: {test_balanced_acc:.4f}")
        else:
            print(f"No valid checkpoint found for iteration {iteration + 1}")
    
    final_model_path = os.path.join(experiment_paths['checkpoints_dir'], f"best_model_iter_{args.iterations}.pth")
    if all_results and os.path.exists(final_model_path):
        # Calculate Inference Time
        final_model = ProposedNet(num_classes=num_classes).to(device)
        final_model.load_state_dict(torch.load(final_model_path, map_location=device))
        inference_speed = measure_inference_speed(final_model)
        
        results_df = pd.DataFrame(all_results)
        summary = {
            "Model": args.model, "Dataset": args.dataset, "Blocks Cfg": f"{model_blocks_cfg}",
            "Mean Accuracy": f"{results_df['acc'].mean():.4f}", "Std Accuracy": f"{results_df['acc'].std():.4f}",
            "Mean Balanced Accuracy": f"{results_df['bacc'].mean():.4f}", "Std Balanced Accuracy": f"{results_df['bacc'].std():.4f}",
            "Parameters (M)": f"{model_complexity.get('params_M', -1):.2f}",
            "FLOPs (G)": f"{model_complexity.get('flops_G', -1):.2f}",
            "GPU Latency (ms)": f"{inference_speed.get('latency_gpu_ms', -1):.2f}",
            "CPU Latency (ms)": f"{inference_speed.get('latency_cpu_ms', -1):.2f}",
            **vars(args)
        }
        
        results_df.to_csv(os.path.join(experiment_paths['results_dir'], 'iteration_results.csv'), index=False)
        with open(os.path.join(experiment_paths['results_dir'], 'summary.txt'), 'w') as f:
            for key, value in summary.items(): f.write(f"{key}: {value}\n")
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        for key, value in summary.items():
            if key in ["Model", "Dataset", "Blocks Cfg", "Mean Accuracy", "Std Accuracy", "Parameters (M)", "FLOPs (G)", "GPU Latency (ms)", "CPU Latency (ms)"]:
                print(f"- {key}: {value}")
        print("="*80)
        print(f"All results saved in: {experiment_paths['experiment_dir']}")

if __name__ == "__main__":
    run_train_test()