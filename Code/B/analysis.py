"""Analysis utilities for Model B (ResNet experiments).

Functions here evaluate the effect of changing the number of training
epochs (training budget) on final test performance. All functions are
typed and log progress rather than printing.
"""

from typing import List, Tuple, Any
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from Code.B.model_b import ModelB

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze_epochs_impact(
    train_loader: Any, val_loader: Any, test_loader: Any
) -> Tuple[List[int], List[float]]:
    """Analyze how number of training epochs affects ResNet performance.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.

    Returns:
        Tuple[List[int], List[float]]: epoch counts tested and corresponding
            test accuracies.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_counts: List[int] = [5, 10, 15, 20, 25]
    test_accs: List[float] = []

    for n_epochs in epoch_counts:
        logger.info("Training for %d epochs...", n_epochs)
        model = ModelB().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Test evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total if total > 0 else 0.0
        test_accs.append(test_acc)
        logger.info("Test Accuracy after %d epochs: %.4f", n_epochs, test_acc)

    # Plot results for reporting
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_counts, test_accs, "o-", linewidth=2, markersize=8)
    plt.xlabel("Number of Epochs (Training Budget)", fontsize=11)
    plt.ylabel("Test Accuracy", fontsize=11)
    plt.title("ResNet: Training Budget (Epochs) vs Performance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "resnet_epochs_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved resnet epochs analysis to %s", out_path)

    return epoch_counts, test_accs
