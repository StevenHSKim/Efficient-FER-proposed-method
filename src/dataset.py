import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# dataset classes
class CKPlusDataset(data.Dataset):
    def __init__(self, data_path, phase, indices, transform=None):
        self.transform = transform
        label_file = os.path.join(data_path, 'ckplus_labels.csv')
        df = pd.read_csv(label_file, sep=',', header=0)
        self.file_names = df['image_name'].values[indices]
        self.labels = df['label'].values[indices]
        self.file_paths = [os.path.join(data_path, 'ckplus_images', f) for f in self.file_names]

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

class ExpWDataset(data.Dataset):
    def __init__(self, data_path, phase, transform=None): # 원본 시그니처
        self.transform = transform
        self.images, self.labels = [], []
        label_file = os.path.join(data_path, 'label/label.lst')
        image_dir = os.path.join(data_path, 'aligned_image')
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    self.images.append(image_name)
                    self.labels.append(int(parts[-1]))
        self.file_paths = [os.path.join(image_dir, f) for f in self.images]

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

class FER2013Dataset(data.Dataset):
    def __init__(self, csv_path, indices, transform=None):
        self.transform = transform
        df = pd.read_csv(csv_path)
        if indices is not None: df = df.iloc[indices]
        self.pixels = df['pixels'].values
        self.labels = df['emotion'].values.astype(np.int64)
        del df

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        pixel_str = self.pixels[idx]
        pixel_array = np.fromstring(pixel_str, dtype=np.uint8, sep=' ').reshape(48, 48)
        image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label
    
class FERPlusDataset(data.Dataset):
    # --- 원본 로직 유지 ---
    def __init__(self, data_path, phase, indices, transform=None):
        self.transform = transform
        df = pd.read_csv(os.path.join(data_path, 'FERPlus_Label_modified.csv'), header=0, dtype={'Image name': str, 'label': int})
        indices = np.array(indices)[np.array(indices) < len(df)]
        self.file_names = df['Image name'].values[indices]
        self.labels = df['label'].values[indices]
        self.file_paths = [os.path.join(data_path, 'FERPlus_Image', f) for f in self.file_names]

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

class RafDBDataset(data.Dataset):
    # --- 원본 로직 유지 ---
    def __init__(self, data_path, phase, indices, transform=None):
        self.transform = transform
        label_file = os.path.join(data_path, 'EmoLabel/list_patition_label.txt')
        df = pd.read_csv(label_file, sep=' ', header=None, names=['name', 'label'])
        self.file_names = df['name'].values[indices]
        self.labels = df['label'].values[indices] - 1
        self.file_paths = [os.path.join(data_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in self.file_names]

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

class SFEWDataset(data.Dataset):
    # --- 원본 로직 유지 ---
    def __init__(self, data_path, phase, indices, transform=None):
        self.transform = transform
        label_path = os.path.join(data_path, 'sfew_2.0_labels.csv')
        df = pd.read_csv(label_path)
        image_names, labels = df['image_name'].values, df['label'].values
        indices = np.array(indices)[np.array(indices) < len(image_names)]
        self.file_names = image_names[indices]
        self.labels = labels[indices]
        self.file_paths = [os.path.join(data_path, 'sfew2.0_images', f) for f in self.file_names]

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label


# dataloaders
# --- 함수 시그니처 변경 (Req #2) (args -> dataset_name, data_path) ---
def load_dataset_info(dataset_name, data_path):
    # (내용은 원본과 동일, args.dataset 대신 dataset_name 사용)
    if dataset_name == 'ckplus':
        df = pd.read_csv(os.path.join(data_path, 'CKPlus/ckplus_labels.csv'), header=0)
        all_data_indices = df.index.values
        all_labels = df['label'].values
        num_classes = len(df['label'].unique())
        use_stratify = False
        
    elif dataset_name == 'expw':
        # ExpW는 원본 방식(Subset)을 위해 전체 데이터셋을 미리 로드해야 함
        full_dataset_for_split = ExpWDataset(os.path.join(data_path, 'ExpW'), phase='all')
        all_data_indices = np.arange(len(full_dataset_for_split))
        all_labels = np.array(full_dataset_for_split.labels)
        num_classes = len(np.unique(all_labels))
        use_stratify = True

    elif dataset_name == 'fer2013':
        df = pd.read_csv(os.path.join(data_path, 'FER2013/fer2013_modified.csv'))
        # all_data_indices = np.array([np.fromstring(p, dtype=int, sep=' ') for p in df['pixels']])
        all_data_indices = df.index.values
        all_labels = df['emotion'].values
        num_classes = len(np.unique(all_labels))
        use_stratify = False

    elif dataset_name == 'ferplus':
        df = pd.read_csv(os.path.join(data_path, 'FERPlus/FERPlus_Label_modified.csv'), header=0)
        if df['label'].min() > 0: df['label'] = df['label'] - 1
        all_data_indices = df.index.values
        all_labels = df['label'].values
        num_classes = len(df['label'].unique())
        use_stratify = False

    elif dataset_name == 'rafdb':
        df = pd.read_csv(os.path.join(data_path, 'raf-basic/EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        all_data_indices = df.index.values
        all_labels = pd.read_csv(os.path.join(data_path, 'raf-basic/EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])['label'].values - 1
        num_classes = 7
        use_stratify = False

    elif dataset_name == 'sfew':
        df = pd.read_csv(os.path.join(data_path, 'SFEW2.0/sfew_2.0_labels.csv'))
        if df['label'].min() > 0: df['label'] = df['label'] - 1
        all_data_indices = df.index.values
        all_labels = df['label'].values
        num_classes = len(df['label'].unique())
        use_stratify = False
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return all_data_indices, all_labels, num_classes, use_stratify


# 통일된 datatransforms
def get_transforms(model_name: str):
    input_size_hw = (112, 112)
    data_transforms = transforms.Compose([
        transforms.Resize(input_size_hw), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))]) 
    val_transforms = transforms.Compose([
        transforms.Resize(input_size_hw), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
    return data_transforms, val_transforms, input_size_hw

# split train/val/test sets
def create_datasets(args, train_val_indices, test_indices, all_data_indices, all_labels, use_stratify, iteration, data_transforms, val_transforms):
    
    stratify_labels = all_labels[train_val_indices] if use_stratify else None
    train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration, stratify=stratify_labels)

    if args.dataset == 'ckplus':
        dataset_path = os.path.join(args.data_path, 'CKPlus')
        train_dataset = CKPlusDataset(dataset_path, 'train', train_indices, data_transforms)
        val_dataset = CKPlusDataset(dataset_path, 'validation', val_indices, val_transforms)
        test_dataset = CKPlusDataset(dataset_path, 'test', test_indices, val_transforms)

    elif args.dataset == 'expw':
        dataset_path = os.path.join(args.data_path, 'ExpW')    
        full_train_dataset = ExpWDataset(dataset_path, 'train', transform=data_transforms)
        full_val_dataset = ExpWDataset(dataset_path, 'val', transform=val_transforms)       
        train_dataset = data.Subset(full_train_dataset, train_indices)
        val_dataset = data.Subset(full_val_dataset, val_indices)
        test_dataset = data.Subset(full_val_dataset, test_indices)

    elif args.dataset == 'fer2013':
        # 수정: 픽셀 배열(all_data_indices[...]) 대신 인덱스를 전달
        csv_path = os.path.join(args.data_path, 'FER2013/fer2013_modified.csv')        
        train_dataset = FER2013Dataset(csv_path, train_indices, transform=data_transforms)
        val_dataset = FER2013Dataset(csv_path, val_indices, transform=val_transforms)
        test_dataset = FER2013Dataset(csv_path, test_indices, transform=val_transforms)
        
    elif args.dataset == 'ferplus':
        dataset_path = os.path.join(args.data_path, 'FERPlus')
        train_dataset = FERPlusDataset(dataset_path, 'train', train_indices, data_transforms)
        val_dataset = FERPlusDataset(dataset_path, 'validation', val_indices, val_transforms)
        test_dataset = FERPlusDataset(dataset_path, 'test', test_indices, val_transforms)
        
    elif args.dataset == 'rafdb':
        dataset_path = os.path.join(args.data_path, 'raf-basic')
        train_dataset = RafDBDataset(dataset_path, 'train', train_indices, data_transforms)
        val_dataset = RafDBDataset(dataset_path, 'validation', val_indices, val_transforms)
        test_dataset = RafDBDataset(dataset_path, 'test', test_indices, val_transforms)
        
    elif args.dataset == 'sfew':
        dataset_path = os.path.join(args.data_path, 'SFEW2.0')
        train_dataset = SFEWDataset(dataset_path, 'train', train_indices, data_transforms)
        val_dataset = SFEWDataset(dataset_path, 'validation', val_indices, val_transforms)
        test_dataset = SFEWDataset(dataset_path, 'test', test_indices, val_transforms)
        
    return train_dataset, val_dataset, test_dataset