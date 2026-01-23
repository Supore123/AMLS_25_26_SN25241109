"""ResNet utilities and training helpers for Model B experiments.

This module defines a ResNet-based model adapted to grayscale input and
several training utilities that return structured training history. Logging
is used instead of print statements and all functions include type hints
and Google-style docstrings.
"""

from typing import Any, Dict, Tuple, List
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- 1. The Model Class (Kept simple) ---
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        # Load ResNet18
        self.net = resnet18(weights=None)
        # Modify for 1-channel input (Grayscale)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        # Modify for Binary Classification
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.net(x)

# --- 2. Training Functions (MOVED OUTSIDE THE CLASS) ---

def train_resnet_with_tracking(
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    epochs: int = 15,
    lr: float = 0.001,
    use_augmentation: bool = True,
) -> Tuple[ModelB, Dict[str, list], float, np.ndarray, np.ndarray, np.ndarray]:
    """Train ResNet and track complete training history.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        use_augmentation (bool): Informational flag; pipeline should
            configure augmentation externally.

    Returns:
        Tuple containing: trained model, history dict, test accuracy and
        arrays (y_true, y_pred, y_probs).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelB().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": [],
    }

    logger.info("Training ResNet (lr=%s, augment=%s)", lr, use_augmentation)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()

        if (epoch + 1) % 3 == 0:
            logger.info(
                "Epoch %d/%d: Train Loss=%.4f, Val Acc=%.4f",
                epoch + 1,
                epochs,
                history["train_loss"][-1],
                history["val_acc"][-1],
            )

    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    y_true_list = []
    y_pred_list = []
    y_probs_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
            y_probs_list.extend(probs.cpu().numpy())

    test_acc = test_correct / test_total if test_total > 0 else 0.0
    logger.info("Final Test Accuracy: %.4f", test_acc)

    return model, history, test_acc, np.array(y_true_list), np.array(y_pred_list), np.array(y_probs_list)


def plot_training_history(history: Dict[str, list], title: str = "ResNet Training") -> None:
    """Plot training dynamics and save figures for the training history.

    Args:
        history (Dict[str, list]): Training history containing loss, acc and
            learning rate entries.
        title (str): Title used for the saved filename and figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss vs Epoch', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy vs Epoch', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Overfitting analysis
    train_val_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(epochs, train_val_gap, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].fill_between(epochs, 0, train_val_gap, alpha=0.3)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Train - Val Accuracy', fontsize=11)
    axes[1, 1].set_title('Overfitting Gap Analysis', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{title.replace(' ', '_')}_history.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved training history plot for %s to %s", title, out_path)


def train_resnet_with_history(train_loader: Any, val_loader: Any, epochs: int = 15, lr: float = 0.001) -> Tuple[ModelB, Dict[str, list]]:
    """Train ResNet and return full training history for plotting.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.

    Returns:
        Tuple[ModelB, Dict[str, list]]: Trained model and history dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelB().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Record history
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)

        logger.info(
            "Epoch %d/%d: Train Loss: %.4f, Val Loss: %.4f, Val Acc: %.4f",
            epoch + 1,
            epochs,
            history["train_loss"][-1],
            history["val_loss"][-1],
            history["val_acc"][-1],
        )

    return model, history
