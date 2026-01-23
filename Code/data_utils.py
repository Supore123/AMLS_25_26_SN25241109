"""
Data utilities for loading BreastMNIST dataset
Handles both classical ML (numpy arrays) and deep learning (PyTorch DataLoaders)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import medmnist
from medmnist import INFO

def get_dataloaders(task='A', augment=False, batch_size=64):
    """
    Load BreastMNIST data for either Task A (classical ML) or Task B (deep learning)
    
    Args:
        task: 'A' for numpy arrays (SVM, etc), 'B' for PyTorch DataLoaders (ResNet)
        augment: Whether to apply data augmentation (only for task B)
        batch_size: Batch size for DataLoaders (only for task B)
    
    Returns:
        For task A: X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays)
        For task B: train_loader, val_loader, test_loader (DataLoaders)
    """
    
    # Download BreastMNIST
    data_flag = 'breastmnist'
    download = True
    
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    if task == 'A':
        # Classical ML - return numpy arrays
        train_dataset = DataClass(split='train', download=download)
        val_dataset = DataClass(split='val', download=download)
        test_dataset = DataClass(split='test', download=download)
        
        # Extract numpy arrays
        X_train = train_dataset.imgs.astype(np.float32) / 255.0
        y_train = train_dataset.labels.squeeze().astype(np.int64)
        
        X_val = val_dataset.imgs.astype(np.float32) / 255.0
        y_val = val_dataset.labels.squeeze().astype(np.int64)
        
        X_test = test_dataset.imgs.astype(np.float32) / 255.0
        y_test = test_dataset.labels.squeeze().astype(np.int64)
        
        print(f"Task A Data Loaded:")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    elif task == 'B':
        # Deep Learning - return DataLoaders
        if augment:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        train_dataset = DataClass(split='train', transform=train_transform, download=download)
        val_dataset = DataClass(split='val', transform=test_transform, download=download)
        test_dataset = DataClass(split='test', transform=test_transform, download=download)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Task B DataLoaders Created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        print(f"  Augmentation: {augment}")
        
        return train_loader, val_loader, test_loader
    
    else:
        raise ValueError("task must be 'A' or 'B'")
