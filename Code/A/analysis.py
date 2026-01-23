"""Analysis helpers for Model A (classical SVM experiments).

This module provides functions to analyze model complexity and training
budget effects for the SVM pipeline. Functions are typed and use logging
instead of printing.
"""

from typing import Tuple, List
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from Code.A.model_a import ModelA

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze_svm_complexity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_hog: bool = True,
) -> Tuple[List[float], List[float], List[float]]:
    """Analyze how SVM complexity (C parameter) affects performance.

    Args:
        X_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.
        use_hog (bool): Whether to extract HOG features before training.

    Returns:
        Tuple[List[float], List[float], List[float]]: C values, training
            accuracies and test accuracies.
    """
    model_instance = ModelA(use_hog=use_hog)
    X_tr_feat = model_instance._extract_features(X_train)
    X_test_feat = model_instance._extract_features(X_test)

    # Normalize features for SVM training
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    C_values = [0.01, 0.1, 1, 10, 100, 1000]
    train_accs: List[float] = []
    test_accs: List[float] = []

    for C in C_values:
        clf = SVC(C=C, kernel="rbf", gamma="scale", random_state=42)
        clf.fit(X_tr_scaled, y_train)

        train_acc = clf.score(X_tr_scaled, y_train)
        test_acc = clf.score(X_test_scaled, y_test)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        logger.info("C=%s: Train=%.4f, Test=%.4f", C, train_acc, test_acc)

    # Plot results for the report
    plt.figure(figsize=(8, 5))
    plt.semilogx(C_values, train_accs, "o-", label="Training Accuracy", linewidth=2)
    plt.semilogx(C_values, test_accs, "s-", label="Test Accuracy", linewidth=2)
    plt.xlabel("C (Regularization Parameter - Model Complexity)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("SVM: Model Complexity vs Performance\n(Gap indicates overfitting)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "svm_complexity_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    return C_values, train_accs, test_accs


def analyze_training_budget_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_hog: bool = True,
) -> Tuple[List[float], List[float]]:
    """Analyze how the training set size affects SVM performance.

    Args:
        X_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.
        use_hog (bool): Whether to extract HOG features before training.

    Returns:
        Tuple[List[float], List[float]]: sample fractions and corresponding test
            accuracies.
    """
    model_instance = ModelA(use_hog=use_hog)
    X_tr_feat = model_instance._extract_features(X_train)
    X_test_feat = model_instance._extract_features(X_test)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # Use a range of sample fractions of the training data
    sample_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    test_accs: List[float] = []

    for size in sample_sizes:
        n_samples = int(len(X_tr_scaled) * size)
        indices = np.random.choice(len(X_tr_scaled), n_samples, replace=False)

        X_subset = X_tr_scaled[indices]
        y_subset = y_train[indices]

        clf = SVC(C=10, kernel="rbf", gamma="scale", random_state=42)
        clf.fit(X_subset, y_subset)

        test_acc = clf.score(X_test_scaled, y_test)
        test_accs.append(test_acc)
        logger.info(
            "%d%% of data (%d samples): Test Acc = %.4f", int(size * 100), n_samples, test_acc
        )

    plt.figure(figsize=(8, 5))
    plt.plot([s * 100 for s in sample_sizes], test_accs, "o-", linewidth=2, markersize=8)
    plt.xlabel("Training Set Size (%)", fontsize=11)
    plt.ylabel("Test Accuracy", fontsize=11)
    plt.title("SVM: Training Budget (Sample Size) vs Performance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "svm_training_budget.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    return sample_sizes, test_accs
