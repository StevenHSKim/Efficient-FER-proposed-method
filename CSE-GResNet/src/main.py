import os
import sys
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")
from models.model import CSE_GResNet
from .dataset import load_dataset_info, get_transforms, create_datasets
from .utils import control_random_seed, measure_model_complexity, measure_inference_speed, THOP_AVAILABLE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['rafdb', 'ferplus', 'fer2013', 'expw', 'sfew', 'ckplus'], help='Dataset to use.')
    parser.add_argument('--data_path', type=str, default='/workspace/datasets', help='Root directory of all datasets.')
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD (if optimizer is sgd)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of train_val data to use for validation')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations (e.g., 10 for 10 runs)')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience.')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name (for compatibility)')
    return parser.parse_args()


def train(train_loader, val_loader, model, criterion, optimizer, scheduler, device, 
          epochs, patience, iteration, checkpoint_dir, dataset_name):
    """
    단일 iteration에 대한 모델 훈련 및 검증 루프
    """
    best_loss, best_acc, patience_counter = float('inf'), 0, 0
    best_checkpoint_path = None
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"Training Iteration {iteration}"):
        running_loss, correct_sum, iter_cnt = 0.0, 0, 0
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for (imgs, targets) in train_pbar:
            iter_cnt += 1; optimizer.zero_grad()
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs); loss = criterion(out, targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct_num.float()/imgs.size(0):.4f}'})
        
        acc = correct_sum.float() / (len(train_loader.dataset) + 1e-6)
        running_loss /= (iter_cnt + 1e-6)
        tqdm.write(f'[Epoch {epoch}] Training accuracy: {acc:.4f}. Loss: {running_loss:.3f}. LR {optimizer.param_groups[0]["lr"]:.6f}')
        
        with torch.no_grad():
            val_loss, iter_cnt, bingo_cnt, sample_cnt = 0.0, 0, 0, 0
            y_true, y_pred = [], []
            model.eval()
            for (imgs, targets) in tqdm(val_loader, desc="Validation", leave=False):
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs); loss = criterion(out, targets)
                val_loss += loss.item(); iter_cnt += 1
                _, predicts = torch.max(out, 1)
                bingo_cnt += torch.eq(predicts, targets).sum().cpu()
                sample_cnt += out.size(0)
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())
        
        val_loss /= (iter_cnt + 1e-6)
        if scheduler: scheduler.step()
        
        acc = np.around((bingo_cnt.float() / (sample_cnt + 1e-6)).numpy(), 4)
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)
        tqdm.write(f"[{datetime.now().strftime('%y%m%d_%H%M%S')}] [Epoch {epoch}] Validation accuracy: {acc:.4f}. bacc: {balanced_acc:.4f}. Loss: {val_loss:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            tqdm.write(f"New best accuracy: {best_acc:.4f}")
            
            if not os.path.exists(checkpoint_dir): 
                os.makedirs(checkpoint_dir)
            prefix = f"cse_gresnet_{dataset_name}_iter{iteration}" 
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith(prefix) and (filename.endswith(".pth") or filename.endswith(".png")):
                    try:
                        os.remove(os.path.join(checkpoint_dir, filename))
                    except OSError as e:
                        print(f"Warning: Could not remove old file {filename}: {e}")

            new_checkpoint_path = os.path.join(checkpoint_dir, 
                                                f"{prefix}_epoch{epoch}_acc{acc:.4f}_bacc{balanced_acc:.4f}.pth")
            try:
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'accuracy': acc, 
                    'balanced_accuracy': balanced_acc,
                }, new_checkpoint_path)
                tqdm.write(f'New best model saved at {new_checkpoint_path}')
                best_checkpoint_path = new_checkpoint_path
            except Exception as e:
                tqdm.write(f"Error saving checkpoint: {e}")
        
        if val_loss < best_loss:
            best_loss = val_loss; patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            tqdm.write("Early stopping triggered"); break
            
    return best_checkpoint_path, best_acc

# test 함수
def test(test_loader, model, checkpoint_path, criterion, device, emotion_labels):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return 0.0, 0.0, float('inf'), None

    with torch.no_grad():
        running_loss, iter_cnt, bingo_cnt, sample_cnt = 0.0, 0, 0, 0
        y_true, y_pred = [], []
        model.eval()
        for (imgs, targets) in tqdm(test_loader, desc="Testing"):
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs); loss = criterion(out, targets)
            running_loss += loss.item(); iter_cnt += 1
            _, predicts = torch.max(out, 1)
            bingo_cnt += torch.eq(predicts, targets).sum().cpu()
            sample_cnt += out.size(0)
            y_true.append(targets.cpu().numpy()); y_pred.append(predicts.cpu().numpy())
    
    if iter_cnt == 0 or sample_cnt == 0: return 0.0, 0.0, 0.0, None

    running_loss /= iter_cnt
    acc = np.around((bingo_cnt.float() / sample_cnt).numpy(), 4)
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)
    
    tqdm.write(f"Test accuracy: {acc:.4f}. bacc: {balanced_acc:.4f}. Loss: {running_loss:.3f}")
    return acc, balanced_acc, running_loss, None


def run_train_test():
    code_start_time = datetime.now().strftime("%y%m%d_%H%M%S") 
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, PyTorch: {torch.__version__}, Thop: {THOP_AVAILABLE}")

    # --- 1. 데이터셋 정보 로드 ---
    try:
        all_data_indices, all_labels, num_classes, use_stratify = load_dataset_info(
            args.dataset, args.data_path
        )
        print(f"Successfully loaded {args.dataset} data. Total samples: {len(all_labels)}. Found {num_classes} classes.")
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        return

    emotion_labels_map = {
        'rafdb': ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'],
        'ckplus': ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'],
        'ferplus': ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'],
        'sfew': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'],
        'expw': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
        'fer2013': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    }
    emotion_labels = emotion_labels_map.get(args.dataset, None)


    # --- 2. 데이터 변환 정의 ---
    data_transforms, val_transforms, input_size_hw = get_transforms(model_name='CSE-GResNet')
    input_size_chw = (3, input_size_hw[0], input_size_hw[1])

    results = []
    checkpoint_dir = f'/workspace/checkpoints/{args.dataset}' 
    if not os.path.exists(checkpoint_dir): 
        os.makedirs(checkpoint_dir)
    
    model_complexity = None
    model_name_log = "" # 로깅용 모델 이름

    # --- 3. Iteration 루프 ---
    for iteration in range(1, args.iterations + 1):
        
        training_start_time = datetime.now().strftime("%y%m%d_%H%M%S") 
        print(f"\n{'='*20} Iteration {iteration}/{args.iterations} (Seed: {iteration}) {'='*20}")
        
        # --- 3-1. 시드 고정  ---
        control_random_seed(iteration)

        # --- 3-2. 데이터 분할 ---
        stratify_array = all_labels if use_stratify else None
        train_val_indices, test_indices = train_test_split(
            all_data_indices, 
            test_size=args.test_size, 
            random_state=iteration, 
            stratify=stratify_array
        )
        
        # --- 3-3. 모델 생성 (Factory 호출) ---
        # 팩토리 함수가 (model, model_name) 튜플 반환
        model, model_name_log = CSE_GResNet(num_classes=num_classes)
        model.to(device)

        # --- 3-4. 모델 복잡도 ---
        if iteration == 1 and model_complexity is None:
            print("\n" + "="*50 + "\nMODEL COMPLEXITY\n" + "="*50)
            model_complexity = measure_model_complexity(model, input_size=input_size_chw, device=device)
            print(f"Model: {model_name_log}")
            print(f"Input size: {input_size_chw}")
            print(f"Total Parameters: {model_complexity['params_M']:.3f}M")
            if THOP_AVAILABLE: print(f"FLOPs: {model_complexity['flops_G']:.3f}G")
            else: print("FLOPs: Not calculated (thop library not available).")
            print("="*50 + "\n")

        # 3-5. 데이터셋 생성 
        try:
            train_dataset, val_dataset, test_dataset = create_datasets(
                args, train_val_indices, test_indices, all_data_indices, all_labels, 
                use_stratify, iteration, data_transforms, val_transforms
            )
            print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
        except Exception as e:
            print(f"Error creating datasets for iteration {iteration}: {e}. Skipping.")
            continue
            
        # --- 3-6. 데이터 로더 ---
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        # --- 3-7. 옵티마이저/스케줄러 ---
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'adam': optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw': optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd': optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # 스케줄러
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # --- 3-8. train ---
        best_checkpoint_path, best_acc = train(
            train_loader, val_loader, model, criterion, optimizer, scheduler, device, 
            args.epochs, args.early_stopping_patience, iteration, 
            checkpoint_dir, args.dataset
        )

        # --- 3-9. 테스트  ---
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            # 위에서 정의한 emotion_labels를 test 함수에 전달
            test_acc, test_balanced_acc, test_loss, cm_path = test(
                test_loader, model, best_checkpoint_path, criterion, device, emotion_labels
            )
            
            latency_gpu_ms, latency_cpu_ms = -1.0, -1.0
            try:
                inference_metrics = measure_inference_speed(model, input_size_chw) 
                latency_gpu_ms = inference_metrics.get('latency_gpu_ms', -1.0)
                latency_cpu_ms = inference_metrics.get('latency_cpu_ms', -1.0)
                print(f"Iteration {iteration} Inference Speed: GPU {latency_gpu_ms:.2f}ms | CPU {latency_cpu_ms:.2f}ms")
            except Exception as e:
                print(f"Warning: Could not measure inference speed for iter {iteration}. Error: {e}")
        else:
            print(f"No valid checkpoint found for iteration {iteration}")
            test_acc, test_balanced_acc, test_loss, cm_path = 0.0, 0.0, 0.0, None
            latency_gpu_ms, latency_cpu_ms = -1.0, -1.0

        # --- 3-10. 로깅 ---
        result_data = {
            "code_start_time": code_start_time, "training_start_time": training_start_time,
            "model_name": model_name_log, "dataset_name": args.dataset, "iteration_seed": iteration,
            "test_accuracy": test_acc, "test_balanced_accuracy": test_balanced_acc,
            "test_loss": test_loss,
            "latency_gpu_ms": latency_gpu_ms, "latency_cpu_ms": latency_cpu_ms,
            "avg_epoch_time_min": -1.0, # Not measured
            "params_M": model_complexity.get('params_M', -1) if model_complexity else -1, 
            "flops_G": model_complexity.get('flops_G', -1) if model_complexity else -1,
            "best_val_acc": best_acc, 
            "checkpoint_path": best_checkpoint_path, "cm_path": cm_path,
            **vars(args) 
        }
        results.append(result_data)

    # --- 4. 최종 결과 요약 ---
    print("\n" + "="*80 + "\nFINAL RESULTS SUMMARY\n" + "="*80)
    if results:
        results_df = pd.DataFrame(results)
        summary = results_df.groupby(['dataset_name', 'model_name'])[
            ['test_accuracy', 'test_balanced_accuracy', 'latency_gpu_ms', 'latency_cpu_ms']
        ].mean()
        print(summary)
        print("\nDetailed Iteration Statistics (Mean ± Std):")
        print(f"Test Accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
        print(f"Test B-Accuracy: {results_df['test_balanced_accuracy'].mean():.4f} ± {results_df['test_balanced_accuracy'].std():.4f}")
        print("="*80)
        
        final_csv_path = os.path.join(checkpoint_dir, f'all_results_{args.dataset}_{model_name_log}_{code_start_time}.csv')
        results_df.to_csv(final_csv_path, index=False)
        print(f"All results saved to: {final_csv_path}")
    else:
        print("No results were generated.")
    
    print(f"Experiment run {code_start_time} finished.")


if __name__ == "__main__":
    run_train_test()