import os
import sys 
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import time 
from datetime import datetime 
import importlib 
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.utils.data as data
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split 
import matplotlib
matplotlib.use('Agg')
eps = sys.float_info.epsilon

from .dataset import load_dataset_info, create_datasets, get_transforms
from .utils import create_experiment_folder, measure_model_complexity, measure_inference_speed, control_random_seed, THOP_AVAILABLE
from .models.AdaDF.adadf_strategy import AdaDFStrategy
from .models.DAN.dan_strategy import DANStrategy
from .models.base.model_strategy import get_model_strategy

from .models.ProposedNet.model import ProposedNet
from .models.DAN.dan import DAN
from .models.POSTERV2.PosterV2_7cls import pyramid_trans_expr2
from .models.AdaDF.model import create_model as create_adadf_model


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Facial Emotion Recognition Training Script")

    # --- 통합 인수 ---
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use (e.g., ckplus ferplus).')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to use (e.g., ProposedNet DAN POSTER AdaDF).')
    parser.add_argument('--iterations', type=int, nargs=2, required=True, metavar=('START', 'END'), help='Iteration range (inclusive) e.g., 1 5 or 3 3.')
    parser.add_argument('--model_dir', type=str, default='src.models', help='Python path to models directory (e.g., src.models)')
    parser.add_argument('--data_path', type=str, default='./datasets', help='Path to the dataset directory.')
    parser.add_argument('--gpu', type=str, default='0', help='Assign GPU(s) by ID(s) (e.g., 0 or 0,1).')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing') 
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of training data to use for validation') 
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience.')
    parser.add_argument('--exp_name', type=str, default='', help='Custom experiment name (optional)')
    parser.add_argument('--seed', default=None, type=int, help='Global random seed.') # Ada-DF에서 사용

    # --- DAN 전용 ---
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head (for DAN).')
    
    # --- Ada-DF 전용 ---
    parser.add_argument('--threshold', default=0.7, type=float, help='Ada-DF threshold')
    parser.add_argument('--sharpen', default=False, type=bool, help='Ada-DF sharpen LD')
    parser.add_argument('--T', default=1.2, type=float, help='Ada-DF Temperature for sharpen')
    parser.add_argument('--alpha', default=None, type=float, help='Ada-DF alpha (None for dynamic)')
    parser.add_argument('--beta', default=3, type=int, help='Ada-DF beta for dynamic alpha')
    parser.add_argument('--max_weight', default=1.0, type=float, help='Ada-DF max attention weight')
    parser.add_argument('--min_weight', default=0.2, type=float, help='Ada-DF min attention weight')
    parser.add_argument('--drop_rate', default=0.0, type=float, help='Ada-DF model drop_rate')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Ada-DF label smoothing')
    parser.add_argument('--tops', default=0.7, type=float, help='Ada-DF tops ratio for RR_loss')
    parser.add_argument('--margin_1', default=0.07, type=float, help='Ada-DF margin_1 for RR_loss')

    return parser.parse_args()

# ==================================================================
# 2. 통합 Train 함수
# ==================================================================

def train_unified(args, train_loader, val_loader, model, strategy, 
                  optimizer, scheduler, device, epochs, patience,
                  iteration, checkpoint_dir, model_name, dataset_name, 
                  training_start_time, results_dir):
    """
    모든 모델을 위한 통합 훈련 함수. strategy 객체가 모델별 특수 로직을 처리.
    """
    best_loss, best_acc, patience_counter = float('inf'), 0, 0
    progress_bar = tqdm(range(1, epochs + 1), 
                        desc=f"Training {model_name}/{dataset_name} Iter {iteration}")
    epoch_times, train_losses, val_losses = [], [], []
    save_path = ""
    
    # Epoch 로그 파일
    log_file_path = os.path.join(results_dir, f"{model_name}_{dataset_name}_iter{iteration}_epoch_log.csv")
    with open(log_file_path, 'w') as f:
        f.write("timestamp,epoch,train_loss,val_loss,val_acc,val_balanced_acc\n")
    
    # AdaDF를 위한 추가 데이터 수집
    collect_outputs = isinstance(strategy, AdaDFStrategy)
    
    for epoch in progress_bar:
        epoch_start_time = time.time()
        strategy.epoch = epoch # AdaDF용
        
        model.train()
        current_train_loss = 0.0
        
        # AdaDF용 출력 수집
        if collect_outputs:
            all_outputs_1, all_targets, all_weights = [], [], []
        
        for (imgs, targets) in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 모델 forward
            model_output = strategy.forward_model(model, imgs)
            
            # Loss 계산 (모델별 전략)
            loss_result = strategy.compute_loss(model_output, targets)
            
            # AdaDF는 추가 정보 반환
            if collect_outputs and isinstance(loss_result, tuple):
                loss, extra_info = loss_result
                all_outputs_1.append(extra_info['outputs_1'])
                all_targets.append(extra_info['targets'])
                all_weights.append(extra_info['attention_weights'])
            else:
                loss = loss_result
            
            loss.backward()
            optimizer.step()
            current_train_loss += loss.item()
        
        if scheduler:
            scheduler.step()
        
        # AdaDF LD 업데이트
        if collect_outputs and all_outputs_1:
            outputs_tensor = torch.cat(all_outputs_1, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            num_classes = outputs_tensor.size(1)
            strategy.update_LD(outputs_tensor, targets_tensor, num_classes)
        
        # Validation
        avg_train_loss = current_train_loss / (len(train_loader) + eps)
        val_loss, acc, balanced_acc = validate_unified(val_loader, model, strategy, device)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 로그 기록
        with open(log_file_path, 'a') as f:
            f.write(f"{current_time},{epoch},{avg_train_loss:.5f},{val_loss:.5f},{acc:.5f},{balanced_acc:.5f}\n")
        
        epoch_end_time = time.time()
        epoch_duration_min = (epoch_end_time - epoch_start_time) / 60
        epoch_times.append(epoch_duration_min)
        progress_bar.set_description(f"Epoch {epoch} | Train: {avg_train_loss:.3f} | Val: {val_loss:.3f} | Acc: {acc:.4f} | BAcc: {balanced_acc:.4f}")

        # 체크포인트 저장 (Best Accuracy 기준)
        if acc > best_acc:
            best_acc = acc
            save_name = f"{training_start_time}_{model_name}_{dataset_name}_iter{iteration}.pth"
            save_path = os.path.join(checkpoint_dir, save_name)
            model_state = (model.module.state_dict()
                           if isinstance(model, nn.DataParallel)
                           else model.state_dict())
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'state_dict': model_state, # 호환성
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'best_acc': best_acc,
            }
            
            # AdaDF용 추가 정보
            if isinstance(strategy, AdaDFStrategy):
                checkpoint['class_distributions'] = strategy.LD.detach()
            
            torch.save(checkpoint, save_path)
        
        # Early stopping (Best Validation Loss 기준)
        if best_loss > val_loss:
            best_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            tqdm.write("Early stopping triggered")
            break
    
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    return save_path, best_acc, avg_epoch_time, train_losses, val_losses


# ==================================================================
# 3. 통합 Validation 함수
# ==================================================================

def validate_unified(val_loader, model, strategy, device):
    """
    모든 모델을 위한 통합 검증 함수. strategy 객체가 모델별 예측 추출을 처리.
    """
    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []
    
    with torch.no_grad():
        for (imgs, targets) in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            # 모델 forward
            model_output = strategy.forward_model(model, imgs)
            
            loss_result = strategy.compute_loss(model_output, targets)
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
            val_loss += loss.item()
            
            # 예측 추출 (모델별 전략)
            predictions_tensor = strategy.get_predictions(model_output)
            _, predicts = torch.max(predictions_tensor, 1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())
    
    if len(val_loader) == 0:
        return 0.0, 0.0, 0.0
    if not y_true or not y_pred:
        return val_loss / (len(val_loader) + eps), 0.0, 0.0
    
    val_loss /= len(val_loader)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        balanced_acc = acc
    
    return val_loss, acc, balanced_acc


# ==================================================================
# 4. 통합 테스트 함수
# ==================================================================

def test_unified(test_loader, model, checkpoint_path, device, strategy):
    """
    모든 모델을 위한 통합 테스트 함수. strategy 객체를 받아 예측에 사용.
    """
    try:
        # 1. 체크포인트 로드
        loaded_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"Loaded checkpoint type: {type(loaded_data)}") 

        state_dict = None
        # 2. State Dict 추출
        if isinstance(loaded_data, dict):
            if 'model_state_dict' in loaded_data:
                state_dict = loaded_data['model_state_dict']
                print("Extracted state_dict from 'model_state_dict' key.")
            elif 'state_dict' in loaded_data: # AdaDF 호환성
                state_dict = loaded_data['state_dict']
                print("Extracted state_dict from 'state_dict' key.")
            else:
                print(f"Error: Checkpoint dictionary does not contain expected state_dict keys. Keys found: {loaded_data.keys()}")
                return 0.0, 0.0, np.array([])
        else:
            state_dict = loaded_data
            print("Loaded state_dict directly (not a dictionary).")

        if state_dict is None:
            print(f"Error: Could not extract state_dict from checkpoint {checkpoint_path}")
            return 0.0, 0.0, np.array([])

        # 3. 모델에 State Dict 로드
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        try:
            load_result = model_to_load.load_state_dict(state_dict, strict=True)
            print(f"Checkpoint load result (strict=True): {load_result}")
        except RuntimeError as e:
            print(f"Error loading state_dict with strict=True: {e}")
            print("Attempting to load with strict=False...")
            load_result = model_to_load.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint load result (strict=False): {load_result}")
            if load_result.missing_keys or load_result.unexpected_keys:
                print("Warning: Checkpoint loaded with strict=False, but there were missing or unexpected keys.")

    except Exception as e:
        print(f"Error during checkpoint loading: {e}")
        return 0.0, 0.0, np.array([])

    # --- 테스트 로직 ---
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for (imgs, targets) in tqdm(test_loader, desc="Testing", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)

            # --- [수정됨] 모델별 예측 출력 분기 ---
            # strategy 객체를 사용하여 예측값 추출
            model_output = strategy.forward_model(model, imgs)
            predictions_tensor = strategy.get_predictions(model_output)
            _, predicts = torch.max(predictions_tensor, 1)
            # ---

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())

    if not y_true or not y_pred: return 0.0, 0.0, np.array([])
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        balanced_acc = acc
    cm = confusion_matrix(y_true, y_pred)
    return acc, balanced_acc, cm

# ==================================================================
# 5. 메인 실행 함수
# ==================================================================

def run_train_test():
    code_start_time = datetime.now().strftime("%y%m%d_%H%M%S") 
    args = parse_args()
    
    # --- GPU 및 DataParallel 설정 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = [int(id) for id in args.gpu.split(',') if id.strip()]
    if torch.cuda.is_available() and device_ids:
        device = torch.device("cuda:0")    
        print(f"Primary device set to: {device}")
        if len(device_ids) > 1:
            print(f"Available GPUs for DataParallel: {device_ids}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    
    torch.backends.cudnn.benchmark = True 
    
    if args.seed is not None:
        control_random_seed(args.seed)

    experiment_paths = create_experiment_folder(args) 
    
    print(f"Starting Experiment Run: {os.path.basename(experiment_paths['experiment_dir'])}")
    print(f" - Datasets: {args.datasets}")
    print(f" - Models: {args.models}")
    print(f" - Iterations: {args.iterations[0]} to {args.iterations[1]}")
    
    if not THOP_AVAILABLE: print("Warning: `thop` is not installed.")

    try:
        model_module = importlib.import_module(args.model_dir.replace('/', '.')) 
    except ImportError:
        print(f"Error: Could not import model module from {args.model_dir}")
        return

    all_run_results = []
    
    for dataset_name in args.datasets:
        print(f"\n{'='*30} Processing Dataset: {dataset_name} {'='*30}")
        
        try:
            all_data_indices, all_labels, num_classes, use_stratify = load_dataset_info(dataset_name, args.data_path)
            print(f"Successfully loaded {dataset_name} data. Total samples: {len(all_labels)}. Found {num_classes} classes.")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}. Skipping.")
            continue
            
        for model_name in args.models:
            print(f"\n{'-'*25} Processing Model: {model_name} {'-'*25}")
            
            # --- 1. 모델 복잡도 계산 (입력 크기 분기) ---
            model_complexity = {"params_M": -1, "flops_G": -1}
            current_input_size = (3, 112, 112)

            try:
                control_random_seed(42) 
                temp_model = None
                if model_name == 'ProposedNet':
                    temp_model = ProposedNet(num_classes=num_classes)
                elif model_name == 'DAN':
                    temp_model = DAN(num_head=args.num_head, num_class=num_classes)
                elif model_name == 'POSTER':
                    temp_model = pyramid_trans_expr2(img_size=current_input_size[1], num_classes=num_classes)
                elif model_name == 'AdaDF':
                    temp_model = create_adadf_model(num_classes, args.drop_rate)
                elif hasattr(model_module, model_name):
                    temp_model = getattr(model_module, model_name)(num_classes=num_classes)
                
                if temp_model:
                    temp_model = temp_model.to(device) 
                    model_complexity = measure_model_complexity(temp_model, input_size=current_input_size)
                    print(f"Model Complexity Report (Input: {current_input_size})")
                    print(f"   - Parameters: {model_complexity['params_M']:.2f}M")
                    if THOP_AVAILABLE: print(f"   - FLOPs: {model_complexity['flops_G']:.2f}G")
                    del temp_model; torch.cuda.empty_cache() 
            except Exception as e:
                print(f"Error calculating complexity for {model_name}: {e}")

            # --- Iteration 루프 ---
            start_iter, end_iter = args.iterations
            for iteration in range(start_iter, end_iter + 1):
                
                training_start_time = datetime.now().strftime("%y%m%d_%H%M%S") 
                print(f"\n--- Iteration {iteration}/{end_iter} (Seed: {iteration}) ---")

                # --- 2. 스플릿 ---
                stratify_array = all_labels if use_stratify else None
                train_val_indices, test_indices = train_test_split(all_data_indices, test_size=args.test_size, random_state=iteration, stratify=stratify_array)
                
                # --- 3. 데이터 변환 ---
                data_transforms, val_transforms, input_size_hw = get_transforms(model_name)
                
                # --- 4. 데이터셋 및 로더 ---
                temp_args = argparse.Namespace(**vars(args)); temp_args.dataset = dataset_name 
                try:
                    train_dataset, val_dataset, test_dataset = create_datasets(temp_args, train_val_indices, test_indices, all_data_indices, all_labels, use_stratify, iteration, data_transforms, val_transforms)
                except Exception as e:
                    print(f"Error creating datasets for iter {iteration}: {e}. Skipping."); continue

                train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
                val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
                test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
                
                # --- 5. 시드 고정 ---
                control_random_seed(iteration)
                
                # --- 6. 모델 생성, 전략 선택, 훈련 실행 ---
                model = None
                strategy = None
                optimizer = None
                scheduler = None
                
                try:
                    # 6.1. 모델 생성
                    if model_name == 'ProposedNet':
                        model = ProposedNet(num_classes=num_classes)
                    elif model_name == 'DAN':
                        model = DAN(num_head=args.num_head, num_class=num_classes)
                    elif model_name == 'POSTER':
                        model = pyramid_trans_expr2(img_size=input_size_hw[0], num_classes=num_classes)
                    elif model_name == 'AdaDF':
                        model = create_adadf_model(num_classes, args.drop_rate)
                    elif hasattr(model_module, model_name):
                         model = getattr(model_module, model_name)(num_classes=num_classes)
                    else:
                        print(f"Error: Model {model_name} not found or not available. Skipping iteration.")
                        continue
                    
                    if len(device_ids) > 1:
                        model = nn.DataParallel(model, device_ids=device_ids)
                    model = model.to(device)
                    
                    # 6.2. 모델별 학습 전략 생성
                    strategy = get_model_strategy(model_name, args, device, num_classes)
                    
                    # 6.3. 옵티마이저 파라미터 설정
                    if isinstance(strategy, DANStrategy):
                        # DAN은 AffinityLoss의 파라미터도 함께 최적화
                        params = list(model.parameters()) + list(strategy.criterion_af.parameters())
                    else:
                        params = model.parameters()

                    # 6.4. 옵티마이저 생성
                    if args.optimizer == 'adam':
                        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
                    elif args.optimizer == 'adamw':
                        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
                    else:
                        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

                    # 6.5. 스케줄러 생성
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                    
                    # 6.6. 통합 훈련 함수 호출
                    best_checkpoint_path, _, avg_epoch_time_min, train_losses, val_losses = train_unified(
                        args, train_loader, val_loader, model, strategy,
                        optimizer, scheduler, device, args.epochs, args.early_stopping_patience,
                        iteration, experiment_paths['checkpoints_dir'], model_name, dataset_name, 
                        training_start_time, experiment_paths['results_dir']
                    )

                except Exception as e:
                    print(f"Error during training/testing of {model_name} (Iter {iteration}): {e}. Skipping.")
                    import traceback
                    traceback.print_exc() # 디버깅을 위한 상세 에러 출력
                    continue
                
                # --- 7. 테스트 ---
                test_acc, test_balanced_acc, cm = 0.0, 0.0, np.array([])
                latency_gpu_ms, latency_cpu_ms = -1.0, -1.0
                
                if os.path.exists(best_checkpoint_path) and strategy is not None:
                    # strategy 객체 전달
                    test_acc, test_balanced_acc, cm = test_unified(test_loader, model, best_checkpoint_path, device, strategy)

                    try:
                        inference_metrics = measure_inference_speed(model, (3, input_size_hw[0], input_size_hw[1]))
                        latency_gpu_ms = inference_metrics.get('latency_gpu_ms', -1.0)
                        latency_cpu_ms = inference_metrics.get('latency_cpu_ms', -1.0)
                        print(f"Iteration {iteration} Inference Speed: GPU {latency_gpu_ms:.2f}ms | CPU {latency_cpu_ms:.2f}ms")
                    except Exception as e:
                        print(f"Warning: Could not measure inference speed for iter {iteration}. Error: {e}")
                
                print(f"Iteration {iteration} Test Acc: {test_acc:.4f}, Test Balanced Acc: {test_balanced_acc:.4f}")
                
                # --- 8. 결과 로깅 ---
                result_data = {
                    "code_start_time": code_start_time, "training_start_time": training_start_time,
                    "model_name": model_name, "dataset_name": dataset_name, "iteration_seed": iteration,
                    "test_accuracy": test_acc, "test_balanced_accuracy": test_balanced_acc,
                    "latency_gpu_ms": latency_gpu_ms,
                    "latency_cpu_ms": latency_cpu_ms,
                    "avg_epoch_time_min": avg_epoch_time_min,
                    "params_M": model_complexity.get('params_M', -1), "flops_G": model_complexity.get('flops_G', -1),
                    "final_train_loss": train_losses[-1] if train_losses else -1,
                    "final_val_loss": val_losses[-1] if val_losses else -1,
                    "best_val_loss": min(val_losses) if val_losses else -1,
                    "checkpoint_path": best_checkpoint_path, "script_name": __file__,
                    **vars(args) 
                }
                all_run_results.append(result_data)
                
                # (메모리 정리를 위해 루프 마지막에 정리)
                del model, strategy, optimizer, scheduler, train_loader, val_loader, test_loader
                torch.cuda.empty_cache()


    # --- 9. 최종 결과 저장 ---
    if all_run_results:
        print("\n" + "="*80 + "\nFINAL RESULTS SUMMARY\n" + "="*80)
        results_df = pd.DataFrame(all_run_results)
        summary = results_df.groupby(['dataset_name', 'model_name'])[['test_accuracy', 'test_balanced_accuracy', 'latency_gpu_ms', 'latency_cpu_ms']].mean()
        print(summary)
        print("="*80)
        final_csv_path = os.path.join(experiment_paths['results_dir'], f'all_results_{code_start_time}.csv')
        results_df.to_csv(final_csv_path, index=False)
        print(f"All results saved to: {final_csv_path}")
    else:
        print("No results were generated.")
    print(f"Experiment run {code_start_time} finished.")


if __name__ == "__main__":
    run_train_test()