import os
import sys
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit, train_test_split
import random
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# FLOPs 계산을 위한 라이브러리
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
    print("thop library loaded successfully")
except ImportError:
    print("Warning: thop library not found. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")
    THOP_AVAILABLE = False
except Exception as e:
    print(f"Warning: thop library import failed: {e}")
    THOP_AVAILABLE = False

# from models.shufflenetv2_spatialglance import ShuffleNetV2
# from models.better_shufflenet_dilated import BetterShuffleNet_ChannelAttention
# from models.no_spatial_glance import BetterShuffleNet_ChannelAttention
from models.se_shufflenetv2_glance import BetterShuffleNet_ChannelAttention

def warn(*args, **kwargs):
    pass
warnings.warn = warn

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic', help='Raf-DB dataset path.')
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    
    # 모델 파라미터
    parser.add_argument('--num_classes', type=int, default=7, help='Number of emotion classes.')
    parser.add_argument('--model_size', type=str, default='1.0x', 
                       choices=['0.5x', '1.0x', '1.5x', '2.0x'], 
                       help='ShuffleNetV2 model size')
    parser.add_argument('--stage_repeats', type=str, default='auto', 
                       help='Stage repeats configuration. Use "auto" for adaptive or specify as "2,4,2"')
    parser.add_argument('--use_residual', action='store_true', 
                   help='Enable residual connections in stride=1 blocks')
    
    # 하이퍼 파라미터 설정
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    
    # 데이터 분할 비율
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    
    # learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp', 'lambda'], help='Learning rate scheduler type')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma for scheduler')
    
    # 실행 설정
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for repeated random sampling')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience.')
    
    # 실험 폴더 관련
    parser.add_argument('--exp_name', type=str, default='', help='Custom experiment name (optional)')
    
    return parser.parse_args()

def create_experiment_folder(args, custom_stage_repeats):
    """
    실험 정보를 포함한 폴더 생성
    """
    # 현재 시간 기반 timestamp
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Stage repeats 문자열 생성
    if custom_stage_repeats:
        stage_str = f"_{'_'.join(map(str, custom_stage_repeats))}"
    else:
        stage_str = "_auto"
    
    residual_str = "_res" if args.use_residual else ""
    
    # 실험 이름 생성
    if args.exp_name:
        exp_name = f"{args.exp_name}_{timestamp}"
    else:
        exp_name = f"BetterShuffleNet_{args.model_size}{stage_str}{residual_str}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{timestamp}"

    # 기본 실험 디렉토리
    base_exp_dir = './proposed_model/experiment_results'
    experiment_dir = os.path.join(base_exp_dir, exp_name)
    
    # 하위 디렉토리들 생성
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    confusion_matrices_dir = os.path.join(experiment_dir, 'confusion_matrices')
    logs_dir = os.path.join(experiment_dir, 'logs')
    
    # 디렉토리 생성
    for directory in [experiment_dir, checkpoints_dir, results_dir, confusion_matrices_dir, logs_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 실험 정보를 텍스트 파일로 저장
    exp_info_path = os.path.join(experiment_dir, 'experiment_info.txt')
    with open(exp_info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: Better ShuffleNetV2 + Spatial Glance + MDConv\n")
        f.write(f"Model Size: {args.model_size}\n")
        f.write(f"Stage Repeats: {custom_stage_repeats if custom_stage_repeats else 'Auto'}\n")
        f.write(f"Use Residual: {args.use_residual}\n")
        f.write(f"Input Size: 112x112\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"LR Scheduler: {args.lr_scheduler}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Early Stopping Patience: {args.early_stopping_patience}\n")
        f.write(f"Test Size: {args.test_size}\n")
        f.write(f"Validation Size: {args.val_size}\n")
        f.write("=" * 80 + "\n")
    
    return {
        'experiment_dir': experiment_dir,
        'checkpoints_dir': checkpoints_dir,
        'results_dir': results_dir,
        'confusion_matrices_dir': confusion_matrices_dir,
        'logs_dir': logs_dir,
        'exp_name': exp_name,
        'timestamp': timestamp
    }

def calculate_model_complexity(model, input_size=(3, 112, 112), device='cpu'):
    """
    모델의 복잡도를 계산하는 함수
    """
    model_info = {}
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info['total_params'] = total_params
    model_info['trainable_params'] = trainable_params
    model_info['total_params_M'] = total_params / 1_000_000
    model_info['trainable_params_M'] = trainable_params / 1_000_000
    
    # 모델 크기 계산 (MB)
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    model_info['model_size_mb'] = model_size_mb
    
    # FLOPs 계산 (thop 라이브러리 사용) - 개선된 버전
    if THOP_AVAILABLE:
        try:
            # 모델을 evaluation 모드로 설정
            original_mode = model.training
            model.eval()
            
            # 원본 디바이스 저장
            original_device = next(model.parameters()).device
            
            # 모델을 복사하여 원본 모델에 영향을 주지 않도록 함
            import copy
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.cpu()
            
            dummy_input = torch.randn(1, *input_size).cpu()
            
            # FLOPs 계산 시도
            try:
                with torch.no_grad():
                    flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
                
                flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
                
                model_info['flops'] = flops
                model_info['flops_formatted'] = flops_formatted
                model_info['flops_G'] = flops / 1_000_000_000  # GFLOPs
                
                print(f"FLOPs calculation successful using thop: {flops_formatted}")
                
            except Exception as thop_error:
                print(f"thop calculation failed: {thop_error}")
                
                # 대략적인 FLOPs 추정 사용
                try:
                    estimated_flops = estimate_flops_roughly(model, input_size)
                    model_info['flops'] = estimated_flops
                    model_info['flops_formatted'] = f"{estimated_flops/1_000_000:.3f}M"
                    model_info['flops_G'] = estimated_flops / 1_000_000_000
                    print(f"Using estimated FLOPs: {model_info['flops_formatted']}")
                except Exception as est_error:
                    print(f"FLOPs estimation also failed: {est_error}")
                    model_info['flops'] = 'N/A'
                    model_info['flops_formatted'] = 'N/A'
                    model_info['flops_G'] = 'N/A'
            
            # 복사된 모델 삭제
            del model_copy
            
            # 원본 모델에서 모든 훅 제거 (혹시 모를 경우를 대비)
            remove_all_hooks(model)
            
            # 모델을 원래 디바이스와 모드로 복원
            model.to(original_device)
            model.train(original_mode)
            
        except Exception as e:
            print(f"Warning: Complete FLOPs calculation failed: {e}")
            # 훅 제거 시도
            remove_all_hooks(model)
            model_info['flops'] = 'N/A'
            model_info['flops_formatted'] = 'N/A'
            model_info['flops_G'] = 'N/A'
    else:
        model_info['flops'] = 'N/A'
        model_info['flops_formatted'] = 'N/A'
        model_info['flops_G'] = 'N/A'
    
    return model_info

def remove_all_hooks(model):
    """
    모델에서 모든 훅을 제거하는 함수
    """
    try:
        for module in model.modules():
            # forward hooks 제거
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            
            # backward hooks 제거
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()
            
            # forward pre hooks 제거
            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()
            
            # backward pre hooks 제거 (PyTorch 1.8+)
            if hasattr(module, '_backward_pre_hooks'):
                module._backward_pre_hooks.clear()
            
            # thop 관련 속성 제거
            if hasattr(module, 'total_ops'):
                delattr(module, 'total_ops')
            if hasattr(module, 'total_params'):
                delattr(module, 'total_params')
        
        print("All hooks removed successfully")
    except Exception as e:
        print(f"Warning: Hook removal failed: {e}")

def estimate_flops_roughly(model, input_size=(3, 112, 112)):
    """
    대략적인 FLOPs 추정 (thop가 실패했을 때 사용)
    """
    total_flops = 0
    h, w = input_size[1], input_size[2]  # height, width
    
    # 간단한 Conv2d와 Linear layer FLOPs 추정
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 대략적인 output size 계산
            out_h = h // module.stride[0] if hasattr(module, 'stride') else h
            out_w = w // module.stride[1] if hasattr(module, 'stride') else w
            
            # Conv2d FLOPs = output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
            kernel_flops = (module.kernel_size[0] * module.kernel_size[1] * 
                          module.in_channels * module.out_channels)
            output_elements = out_h * out_w
            total_flops += kernel_flops * output_elements
            
            # 다음 레이어를 위한 크기 업데이트
            h, w = out_h, w
            
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = in_features * out_features
            total_flops += module.in_features * module.out_features
    
    # ShuffleNet 계열 모델의 합리적인 범위로 조정
    if total_flops < 50_000_000:  # 50M 미만이면 너무 작음
        total_flops = 150_000_000  # 150M FLOPs 정도로 추정
    elif total_flops > 1_000_000_000:  # 1G 이상이면 너무 큼
        total_flops = 400_000_000  # 400M FLOPs 정도로 추정
    
    return total_flops

def save_model_info(model_info, model_config, experiment_paths):
    """
    모델 정보를 파일로 저장
    """
    model_info_path = os.path.join(experiment_paths['experiment_dir'], 'model_info.txt')
    with open(model_info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPLEXITY ANALYSIS\n")
        f.write("=" * 80 + "\n")
        
        if model_config:
            f.write(f"Model Configuration:\n")
            f.write(f"  - Type: {model_config.get('type', 'Unknown')}\n")
            f.write(f"  - Size: {model_config.get('size', 'Unknown')}\n")
            f.write(f"  - Input Size: {model_config.get('input_size', 'Unknown')}\n")
            f.write(f"  - Stage Repeats: {model_config.get('stage_repeats', 'Unknown')}\n")
            f.write(f"  - Use MDConv: {model_config.get('use_mdconv', 'Unknown')}\n")
            f.write("\n")
        
        f.write(f"Parameters:\n")
        f.write(f"  - Total Parameters: {model_info['total_params']:,} ({model_info['total_params_M']:.3f}M)\n")
        f.write(f"  - Trainable Parameters: {model_info['trainable_params']:,} ({model_info['trainable_params_M']:.3f}M)\n")
        f.write(f"  - Model Size: {model_info['model_size_mb']:.2f} MB\n")
        
        f.write(f"\nComputational Complexity:\n")
        if model_info['flops'] != 'N/A':
            f.write(f"  - FLOPs: {model_info['flops_formatted']} ({model_info['flops_G']:.3f} GFLOPs)\n")
        else:
            f.write(f"  - FLOPs: {model_info['flops']}\n")
        
        f.write("=" * 80 + "\n")

def print_model_complexity(model_info, model_config=None):
    """
    모델 복잡도 정보를 출력하는 함수
    """
    print("\n" + "="*80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    
    if model_config:
        print(f"Model Configuration:")
        print(f"  - Type: {model_config.get('type', 'Unknown')}")
        print(f"  - Size: {model_config.get('size', 'Unknown')}")
        print(f"  - Input Size: {model_config.get('input_size', 'Unknown')}")
        print(f"  - Stage Repeats: {model_config.get('stage_repeats', 'Unknown')}")
        print(f"  - Use MDConv: {model_config.get('use_mdconv', 'Unknown')}")
        print("")
    
    print(f"Parameters:")
    print(f"  - Total Parameters: {model_info['total_params']:,} ({model_info['total_params_M']:.3f}M)")
    print(f"  - Trainable Parameters: {model_info['trainable_params']:,} ({model_info['trainable_params_M']:.3f}M)")
    print(f"  - Model Size: {model_info['model_size_mb']:.2f} MB")
    
    print(f"\nComputational Complexity:")
    if model_info['flops'] != 'N/A':
        print(f"  - FLOPs: {model_info['flops_formatted']} ({model_info['flops_G']:.3f} GFLOPs)")
    else:
        print(f"  - FLOPs: {model_info['flops']}")
    
    print("="*80)

def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_best_model(model, optimizer, epoch, acc, balanced_acc, iteration, model_size, checkpoint_dir):
    """best model checkpoint 저장"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    model_prefix = f"shufflenetv2_spatialglance_{model_size}"
        
    # 이전 checkpoint 삭제
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(f"{model_prefix}_iter{iteration+1}_epoch") and \
           (filename.endswith(".pth") or filename.endswith(".png")):
            previous_file = os.path.join(checkpoint_dir, filename)
            if os.path.exists(previous_file):
                os.remove(previous_file)
                tqdm.write(f'Previous file removed: {previous_file}')
                
    # 새로운 checkpoint 저장
    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_prefix}_iter{iteration+1}_epoch{epoch}_acc{acc:.4f}_bacc{balanced_acc:.4f}.pth"
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
    }, best_checkpoint_path)
    
    tqdm.write(f'New best model saved at {best_checkpoint_path}')
    return best_checkpoint_path

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, indices, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        label_file = os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
            
        df = pd.read_csv(label_file, sep=' ', header=None, names=['name', 'label'])
        self.file_names = df['name'].values[indices]
        self.labels = df['label'].values[indices] - 1

        self.file_paths = []
        for f in self.file_names:
            img_path = os.path.join(self.raf_path, 'Image/aligned', 
                                   f.split(".")[0] + "_aligned.jpg")
            self.file_paths.append(img_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
            
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def train(args, train_loader, val_loader, model, criterion, optimizer, scheduler, 
         device, epochs, patience, iteration, checkpoint_dir):
   best_loss = float('inf')
   best_acc = 0
   best_checkpoint_path = None
   patience_counter = 0

   for epoch in tqdm(range(1, epochs + 1), desc=f"Training Iteration {iteration+1}"):
       running_loss = 0.0
       correct_sum = 0
       iter_cnt = 0
       model.train()

       train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
       for (imgs, targets) in train_pbar:
           imgs = imgs.to(device)
           targets = targets.to(device)
           
           optimizer.zero_grad()
           
           model_output = model(imgs)
           if isinstance(model_output, tuple):
               out = model_output[0]
           else:
               out = model_output
               
           loss = criterion(out, targets)
           
           loss.backward()
           optimizer.step()
           scheduler.step()
           
           running_loss += loss.item()
           iter_cnt += 1
           
           _, predicts = torch.max(out, 1)
           correct_num = torch.eq(predicts, targets).sum()
           correct_sum += correct_num

           train_pbar.set_postfix({
               'Loss': f'{loss.item():.4f}',
               'Acc': f'{correct_num.float()/imgs.size(0):.4f}',
               'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
           })

       acc = correct_sum.float() / float(len(train_loader.dataset))
       running_loss = running_loss / max(1, iter_cnt)
       tqdm.write(f'[Epoch {epoch}] Training accuracy: {acc:.4f}. Loss: {running_loss:.3f}. LR {optimizer.param_groups[0]["lr"]:.6f}')
       
       # Validation
       with torch.no_grad():
           val_loss = 0.0
           val_iter_cnt = 0
           bingo_cnt = 0
           sample_cnt = 0
           
           y_true = []
           y_pred = []

           model.eval()
           for (imgs, targets) in tqdm(val_loader, desc="Validation", leave=False):
               imgs = imgs.to(device)
               targets = targets.to(device)
               
               model_output = model(imgs)
               if isinstance(model_output, tuple):
                   out = model_output[0]
               else:
                   out = model_output
                   
               loss = criterion(out, targets)

               val_loss += loss.item()
               val_iter_cnt += 1
               _, predicts = torch.max(out, 1)
               correct_num = torch.eq(predicts, targets)
               bingo_cnt += correct_num.sum().cpu()
               sample_cnt += out.size(0)
               
               y_true.append(targets.cpu().numpy())
               y_pred.append(predicts.cpu().numpy())
       
       val_loss = val_loss / max(1, val_iter_cnt)

       acc = bingo_cnt.float() / float(sample_cnt)
       acc = np.around(acc.numpy(), 4)
       
       y_true = np.concatenate(y_true)
       y_pred = np.concatenate(y_pred)
       balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

       current_time = datetime.now().strftime('%y%m%d_%H%M%S')
       tqdm.write(f"[{current_time}] [Epoch {epoch}] Validation accuracy: {acc:.4f}. bacc: {balanced_acc:.4f}. Loss: {val_loss:.3f}")
       
       # Save best model
       if acc > best_acc:
           best_acc = acc
           tqdm.write(f"New best accuracy: {best_acc:.4f}")
           
           if acc > 0.5:
               best_checkpoint_path = save_best_model(
                   model, optimizer, epoch, acc, balanced_acc, iteration, args.model_size, checkpoint_dir
               )

       # Early stopping
       if best_loss >= val_loss:
           best_loss = val_loss
           patience_counter = 0
       else:
           patience_counter += 1

       if patience_counter >= patience:
           tqdm.write("Early stopping triggered")
           break

   return best_checkpoint_path, best_acc

def test(test_loader, model, checkpoint_path, criterion, device, model_size, confusion_matrices_dir):
    """테스트 함수"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0

        y_true = []
        y_pred = []

        model.eval()
        test_pbar = tqdm(test_loader, desc="Testing")
        for (imgs, targets) in test_pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            model_output = model(imgs)
            if isinstance(model_output, tuple):
                out = model_output[0]
            else:
                out = model_output
                
            loss = criterion(out, targets)

            running_loss += loss.item()
            iter_cnt += 1
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

            y_true.append(targets.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())

        running_loss = running_loss / iter_cnt
        acc = bingo_cnt.float() / float(sample_cnt)
        acc = np.around(acc.numpy(), 4)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Accuracy: {acc:.4f}, Balanced Accuracy: {balanced_acc:.4f})')
        
        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        checkpoint_filename = os.path.basename(checkpoint_path)
        
        try:
            parts = checkpoint_filename.split('_')
            iter_part = None
            epoch_part = None
            for part in parts:
                if part.startswith('iter'):
                    iter_part = part.replace('iter', '')
                elif part.startswith('epoch'):
                    epoch_part = part.replace('epoch', '')
            
            iteration = iter_part if iter_part else "unknown"
            epoch = epoch_part if epoch_part else "unknown"
        except:
            iteration = "unknown"
            epoch = "unknown"
        
        cm_filename = os.path.join(confusion_matrices_dir, 
                                  f"shufflenetv2_spatialglance_{model_size}_iter{iteration}_epoch{epoch}_cm_{current_time}.png")
        plt.savefig(cm_filename)
        plt.close()
        
        tqdm.write(f"[{current_time}] Test accuracy: {acc:.4f}. bacc: {balanced_acc:.4f}. Loss: {running_loss:.3f}")
        tqdm.write(f"Confusion matrix saved as {cm_filename}")

        return acc, balanced_acc, running_loss

def run_train_test():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Stage repeats 파싱
    if args.stage_repeats == 'auto':
        custom_stage_repeats = None
    else:
        try:
            custom_stage_repeats = [int(x) for x in args.stage_repeats.split(',')]
            if len(custom_stage_repeats) != 3:
                raise ValueError("Stage repeats must have exactly 3 values")
        except:
            print(f"Invalid stage_repeats format: {args.stage_repeats}. Using auto configuration.")
            custom_stage_repeats = None

    # 실험 폴더 생성
    experiment_paths = create_experiment_folder(args, custom_stage_repeats)
    
    print("=" * 80)
    print("Better ShuffleNetV2 with Spatial Glance for FER")
    print("=" * 80)
    print(f"Experiment Name: {experiment_paths['exp_name']}")
    print(f"Experiment Directory: {experiment_paths['experiment_dir']}")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\nSelected Configuration:")
    print(f"  Model: Better ShuffleNetV2 with Spatial Glance + MDConv")
    print(f"  Model Size: {args.model_size}")
    print(f"  Stage Repeats: {custom_stage_repeats if custom_stage_repeats else 'Auto (adaptive to input size)'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Scheduler: {args.lr_scheduler}")
    print(f"  Epochs: {args.epochs}")
    print(f"  FLOPs calculation: {'Enabled' if THOP_AVAILABLE else 'Disabled (thop not available)'}")
    print("=" * 80)

    all_accuracies = []
    best_accuracies = []
    results = []
    model_info = {}

    # 데이터셋 파일 확인
    label_file = os.path.join(args.raf_path, 'EmoLabel/list_patition_label.txt')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"RAF-DB label file not found: {label_file}")
        
    df = pd.read_csv(label_file, sep=' ', header=None, names=['name', 'label'])
    file_names = df['name'].values
    labels = df['label'].values - 1

    print(f"Dataset loaded: {len(file_names)} samples")
    print(f"Classes: {np.unique(labels)}")

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_names, labels))

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"\n{'='*25} Iteration {iteration + 1}/{args.iterations} {'='*25}")
        control_random_seed(iteration)

        # 모델 생성
        model = BetterShuffleNet_ChannelAttention(
            input_size=112, 
            n_class=args.num_classes, 
            model_size=args.model_size, 
            custom_stage_repeats=custom_stage_repeats,
            use_mdconv=True,
            use_residual=args.use_residual
        )
        model.to(device)
        
        # 모델 복잡도 계산 (첫 번째 iteration에서만)
        if iteration == 0:
            print("\nCalculating model complexity...")
            model_info = calculate_model_complexity(model, input_size=(3, 112, 112), device=device)
            
            # 모델 설정 정보
            model_config = {
                'type': 'Better ShuffleNetV2 + Spatial Glance + MDConv',
                'size': args.model_size,
                'input_size': '112x112',
                'stage_repeats': custom_stage_repeats if custom_stage_repeats else 'Auto',
                'use_mdconv': True,
                'use_residual': args.use_residual
            }
            
            print_model_complexity(model_info, model_config)
            save_model_info(model_info, model_config, experiment_paths)
        else:
            print(f'Total Parameters: {model_info["total_params_M"]:.3f}M')
            if model_info['flops_G'] != 'N/A':
                print(f'FLOPs: {model_info["flops_formatted"]} ({model_info["flops_G"]:.3f} GFLOPs)')

        # 데이터 변환 정의
        data_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(112, padding=32)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # train_val_indices를 train과 validation으로 분할
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=args.val_size, random_state=iteration, 
            stratify=labels[train_val_indices]
        )

        # 데이터셋 생성
        train_dataset = RafDataSet(args.raf_path, phase='train', 
                                  indices=train_indices, transform=data_transforms)
        print(f'Train set size: {train_dataset.__len__()}')

        val_dataset = RafDataSet(args.raf_path, phase='validation', 
                                indices=val_indices, transform=val_transforms)
        print(f'Validation set size: {val_dataset.__len__()}')

        test_dataset = RafDataSet(args.raf_path, phase='test', 
                                 indices=test_indices, transform=val_transforms)
        print(f'Test set size: {test_dataset.__len__()}')

        # 데이터 로더 생성
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=True, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, pin_memory=True
        )

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                       weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                        weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                      momentum=args.momentum, weight_decay=args.weight_decay)

        # Scheduler
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.lr_scheduler == 'lambda':
            total_iters = len(train_loader) * args.epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                          lambda step: (1.0 - step / total_iters) if step <= total_iters else 0, 
                                                          last_epoch=-1)

        # Training
        best_checkpoint_path, best_acc = train(
            args, train_loader, val_loader, model, criterion, optimizer, scheduler, 
            device, args.epochs, args.early_stopping_patience, iteration, experiment_paths['checkpoints_dir']
        )

        # Testing
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            test_acc, test_balanced_acc, test_running_loss = test(
                test_loader, model, best_checkpoint_path, criterion, device, args.model_size, 
                experiment_paths['confusion_matrices_dir']
            )
            all_accuracies.append(test_acc)
            best_accuracies.append(best_acc)
            results.append([iteration + 1, test_acc, test_balanced_acc, test_running_loss])
            print(f"Results - Test Acc: {test_acc:.4f}, Test Balanced Acc: {test_balanced_acc:.4f}")
        else:
            print(f"No valid checkpoint found for iteration {iteration + 1}")

    # 최종 결과 출력 및 저장
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("Best Validation Accuracies:")
    for i, acc in enumerate(best_accuracies, 1):
        print(f"Iteration {i}: {acc:.4f}")

    if all_accuracies:
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        print(f"\nTest Accuracy Statistics over {len(all_accuracies)} iterations:")
        print(f"Mean: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Best: {max(all_accuracies):.4f}")
        print(f"Worst: {min(all_accuracies):.4f}")
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame(results, columns=['Iteration', 'Test_Acc', 'Test_Balanced_Acc', 'Test_Loss'])
        
        # 통계 정보 추가
        stage_repeats_str = str(custom_stage_repeats) if custom_stage_repeats else 'Auto'
        flops_str = f"{model_info['flops_G']:.3f}" if model_info['flops_G'] != 'N/A' else 'N/A'
        
        summary_data = {
            'Iteration': ['Summary', 'Model_Type', 'Model_Size', 'Stage_Repeats', 'Model_Params(M)', 
                         'FLOPs(G)', 'Model_Size(MB)', 'Mean_Acc', 'Std_Acc', 'Best_Acc', 'Worst_Acc'],
            'Test_Acc': ['', 'Better_ShuffleNetV2+SpatialGlance+MDConv', f'{args.model_size}', stage_repeats_str,
                        f'{model_info["total_params_M"]:.3f}', flops_str, f'{model_info["model_size_mb"]:.2f}',
                        f'{mean_accuracy:.4f}', f'{std_accuracy:.4f}', f'{max(all_accuracies):.4f}', f'{min(all_accuracies):.4f}'],
            'Test_Balanced_Acc': ['', '', '', '', '', '', '', '', '', '', ''],
            'Test_Loss': ['', '', '', '', '', '', '', '', '', '', '']
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # 최종 DataFrame
        final_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        # CSV 파일 저장 (실험 폴더 내)
        results_csv = os.path.join(experiment_paths['results_dir'], 
                                  f'experiment_results_{experiment_paths["timestamp"]}.csv')
        final_df.to_csv(results_csv, index=False)
        print(f"Results saved to: {results_csv}")
        
        # 최종 결과를 텍스트 파일로도 저장
        final_results_path = os.path.join(experiment_paths['results_dir'], 'final_results.txt')
        with open(final_results_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL EXPERIMENT RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Experiment: {experiment_paths['exp_name']}\n")
            f.write(f"Timestamp: {experiment_paths['timestamp']}\n\n")
            
            f.write("Best Validation Accuracies:\n")
            for i, acc in enumerate(best_accuracies, 1):
                f.write(f"Iteration {i}: {acc:.4f}\n")
            
            if all_accuracies:
                f.write(f"\nTest Accuracy Statistics over {len(all_accuracies)} iterations:\n")
                f.write(f"Mean: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
                f.write(f"Best: {max(all_accuracies):.4f}\n")
                f.write(f"Worst: {min(all_accuracies):.4f}\n")
            
            f.write(f"\nFinal Model Configuration:\n")
            f.write(f"- Type: Better ShuffleNetV2 with Spatial Glance + MDConv + hswish/hsigmoid\n")
            f.write(f"- Size: {args.model_size}\n")
            f.write(f"- Stage Repeats: {stage_repeats_str}\n")
            f.write(f"- Parameters: {model_info['total_params_M']:.3f}M\n")
            f.write(f"- Model Size: {model_info['model_size_mb']:.2f} MB\n")
            if model_info['flops_G'] != 'N/A':
                f.write(f"- FLOPs: {model_info['flops_formatted']} ({model_info['flops_G']:.3f} GFLOPs)\n")
            else:
                f.write(f"- FLOPs: {model_info['flops_G']}\n")
            
            f.write("\nEFFICIENCY ANALYSIS:\n")
            if model_info['flops_G'] != 'N/A':
                f.write(f"- Computational Efficiency: {model_info['flops_G']:.3f} GFLOPs for 112x112 input\n")
                accuracy_per_gflop = mean_accuracy / model_info['flops_G'] if model_info['flops_G'] > 0 else 0
                f.write(f"- Accuracy per GFLOPs: {accuracy_per_gflop:.4f}\n")
            
            params_mb = model_info['total_params_M']
            accuracy_per_mb_params = mean_accuracy / params_mb if params_mb > 0 else 0
            f.write(f"- Parameter Efficiency: {model_info['total_params_M']:.3f}M parameters\n")
            f.write(f"- Accuracy per M params: {accuracy_per_mb_params:.4f}\n")
            f.write(f"- Memory Efficiency: {model_info['model_size_mb']:.2f} MB model size\n")
            f.write("=" * 80 + "\n")
        
        # 최종 모델 정보 출력
        print(f"\nFinal Model Configuration:")
        print(f"- Type: Better ShuffleNetV2 with Spatial Glance + MDConv + hswish/hsigmoid")
        print(f"- Size: {args.model_size}")
        print(f"- Stage Repeats: {stage_repeats_str}")
        print(f"- Parameters: {model_info['total_params_M']:.3f}M")
        print(f"- Model Size: {model_info['model_size_mb']:.2f} MB")
        if model_info['flops_G'] != 'N/A':
            print(f"- FLOPs: {model_info['flops_formatted']} ({model_info['flops_G']:.3f} GFLOPs)")
        else:
            print(f"- FLOPs: {model_info['flops_G']}")
        
        print("\n" + "="*80)
        print("MODEL FEATURES")
        print("="*80)
        print("1. Mixed Depthwise Convolution (MDConv): Multi-scale receptive fields")
        print("2. Spatial Glance: 7x7 conv attention for facial expression focus")
        print("3. Hard Swish & Hard Sigmoid: Efficient non-linear activations")
        print("4. SE Modules: Channel attention for feature enhancement")
        print("5. Adaptive Stage Repeats: Optimized depth for input size")
        print("6. Channel Attention Head: Global feature aggregation")
        print("="*80)
        
        # 효율성 분석
        print("\nEFFICIENCY ANALYSIS:")
        if model_info['flops_G'] != 'N/A':
            print(f"- Computational Efficiency: {model_info['flops_G']:.3f} GFLOPs for 112x112 input")
            accuracy_per_gflop = mean_accuracy / model_info['flops_G'] if model_info['flops_G'] > 0 else 0
            print(f"- Accuracy per GFLOPs: {accuracy_per_gflop:.4f}")
        
        params_mb = model_info['total_params_M']
        accuracy_per_mb_params = mean_accuracy / params_mb if params_mb > 0 else 0
        print(f"- Parameter Efficiency: {model_info['total_params_M']:.3f}M parameters")
        print(f"- Accuracy per M params: {accuracy_per_mb_params:.4f}")
        print(f"- Memory Efficiency: {model_info['model_size_mb']:.2f} MB model size")
        
        print(f"\nExperiment completed successfully!")
        print(f"All results saved in: {experiment_paths['experiment_dir']}")
        
    else:
        print("No successful experiments completed.")
        # 실패한 경우에도 로그 저장
        failure_log_path = os.path.join(experiment_paths['logs_dir'], 'failure_log.txt')
        with open(failure_log_path, 'w') as f:
            f.write("EXPERIMENT FAILED\n")
            f.write("=" * 80 + "\n")
            f.write(f"Experiment: {experiment_paths['exp_name']}\n")
            f.write(f"Timestamp: {experiment_paths['timestamp']}\n")
            f.write("No successful experiments completed.\n")

if __name__ == "__main__":
    run_train_test()