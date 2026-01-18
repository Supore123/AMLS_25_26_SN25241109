import medmnist
from medmnist import INFO
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def get_dataloaders(task='B', augment=False, batch_size=64):
    """
    Args:
        task: 'A' for flattened numpy arrays (SVM), 'B' for PyTorch loaders (CNN)
        augment: Boolean, whether to apply data augmentation
    """
    info = INFO['breastmnist']
    # Normalization is critical for both SVM and Neural Networks
    # Mean 0.5, Std 0.5 centers the data around 0
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Augmentation Pipeline (Task Requirement [cite: 163])
    # We use RandomRotation and HorizontalFlip
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        train_transform = base_transform

    # Download=True allows it to work even if Datasets folder is initially empty
    train_dataset = medmnist.BreastMNIST(split='train', transform=train_transform, download=True)
    val_dataset = medmnist.BreastMNIST(split='val', transform=base_transform, download=True)
    test_dataset = medmnist.BreastMNIST(split='test', transform=base_transform, download=True)

    if task == 'A':
        # Return Numpy arrays for Scikit-Learn (Model A)
        # We access the raw .imgs and flattened them for the SVM
        # shape: (N, 28, 28)
        return (train_dataset.imgs, train_dataset.labels.ravel(),
                val_dataset.imgs, val_dataset.labels.ravel(),
                test_dataset.imgs, test_dataset.labels.ravel())
    
    else:
        # Return Loaders for PyTorch (Model B)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Validation and Test should NOT be shuffled
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
