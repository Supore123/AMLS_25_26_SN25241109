"""
AMLS 25/26 Assignment - Benchmarking Study Main Execution Script
Compares SVM and ResNet performance on BreastMNIST dataset.
"""

import sys
import os
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.base import BaseEstimator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration & Setup ---

# Ensure Code directory is in path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code'))

# Define Output Directories
FIGURES_DIR = os.path.join("Code", "output", "figures")
RESULTS_DIR = os.path.join("Code", "output", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure Logging
LOG_PATH = os.path.join(RESULTS_DIR, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding='utf-8')
    ],
)
logger = logging.getLogger(__name__)

# --- Local Imports ---
# Importing inside try/except block or after path setup is standard when project structure is complex
try:
    from data_utils import get_dataloaders
    from A.model_a import ModelA
    from A.analysis import analyze_svm_complexity, analyze_training_budget_svm
    from B.model_b import train_resnet_with_tracking, plot_training_history
    from B.analysis import analyze_epochs_impact
    from visualization import plot_learning_curve
    from evaluation import plot_precision_recall_curve
except ImportError as e:
    logger.critical(f"Failed to import project modules: {e}")
    sys.exit(1)


def analyze_medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Calculates and visualizes medical diagnostic metrics.

    Computes Sensitivity, Specificity, PPV, and NPV. Generates and saves
    a confusion matrix heatmap.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        model_name: Name of the model for display and file naming.

    Returns:
        Dictionary containing calculated metrics and raw confusion matrix counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics with zero-division protection
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    logger.info(f"--- Medical Metrics: {model_name} ---")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
    logger.info(f"Specificity:          {specificity:.4f}")
    logger.info(f"PPV (Precision):      {ppv:.4f}")
    logger.info(f"NPV:                  {npv:.4f}")

    if sensitivity < 0.80:
        logger.warning(
            f"{model_name} has low sensitivity ({sensitivity:.2f}). "
            f"Risk of missing malignant cases."
        )

    # Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'{model_name}\nSensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
    plt.tight_layout()

    filename = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def run_svm_experiments() -> Tuple[Dict[str, float], BaseEstimator, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the experimental pipeline for Model A (SVM).

    Includes feature comparison (Raw vs HOG), learning curve analysis,
    complexity analysis, and training budget analysis.

    Returns:
        Tuple containing:
            - Dictionary of accuracy results.
            - The best trained SVM estimator.
            - Test set ground truth labels.
            - Test set predicted probabilities.
            - Test set binary predictions.
    """
    logger.info("="*40)
    logger.info("Starting Model A (SVM) Experiments")
    logger.info("="*40)

    # Load data for classical ML task
    X_tr, y_tr, X_val, y_val, X_test, y_test = get_dataloaders(task='A')
    results = {}

    # Feature Pipeline Comparison: Raw vs HOG
    logger.info("Experiment: Feature Pipeline Comparison (Raw vs HOG)")

    model_raw = ModelA(use_hog=False)
    _, acc_raw = model_raw.run(X_tr, y_tr, X_val, y_val, X_test, y_test)
    results['SVM_Raw'] = acc_raw

    model_hog = ModelA(use_hog=True)
    best_hog, acc_hog = model_hog.run(X_tr, y_tr, X_val, y_val, X_test, y_test)
    results['SVM_HOG'] = acc_hog

    improvement = (acc_hog - acc_raw) * 100
    logger.info(f"Raw Accuracy: {acc_raw:.4f} | HOG Accuracy: {acc_hog:.4f}")
    logger.info(f"HOG Improvement: {improvement:.2f}%")

    # Learning Curve Analysis
    logger.info("Experiment: Learning Curve Analysis")
    # Extract features manually to pass consistent data to the plotter
    X_hog_features = model_hog._extract_features(X_tr)
    plot_learning_curve(best_hog, X_hog_features, y_tr, "SVM_with_HOG", cv=5)

    # Complexity Analysis
    logger.info("Experiment: Model Complexity (C parameter)")
    analyze_svm_complexity(X_tr, y_tr, X_test, y_test, use_hog=True)

    # Training Budget Analysis
    logger.info("Experiment: Training Budget (Sample Size)")
    analyze_training_budget_svm(X_tr, y_tr, X_test, y_test, use_hog=True)

    # Generate predictions for comparative analysis
    X_test_hog = model_hog._extract_features(X_test)
    y_probs = best_hog.predict_proba(X_test_hog)[:, 1]
    y_pred = best_hog.predict(X_test_hog)

    return results, best_hog, y_test, y_probs, y_pred


def run_resnet_experiments() -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the experimental pipeline for Model B (ResNet).

    Includes data augmentation comparison and training budget (epochs) analysis.

    Returns:
        Tuple containing:
            - Dictionary of accuracy results.
            - Test set ground truth labels (from augmented run).
            - Test set predicted probabilities.
            - Test set binary predictions.
    """
    logger.info("="*40)
    logger.info("Starting Model B (ResNet) Experiments")
    logger.info("="*40)

    results = {}

    # Augmentation Comparison
    logger.info("Experiment: Data Augmentation Impact")

    # Train without augmentation
    logger.info("Training ResNet WITHOUT augmentation...")
    train_no_aug, val_no_aug, test_no_aug = get_dataloaders(task='B', augment=False, batch_size=64)
    _, hist_no_aug, acc_no_aug, _, _, _ = train_resnet_with_tracking(
        train_no_aug, val_no_aug, test_no_aug, epochs=15, lr=0.001, use_augmentation=False
    )
    plot_training_history(hist_no_aug, "ResNet_No_Augmentation")
    results['ResNet_No_Aug'] = acc_no_aug

    # Train with augmentation
    logger.info("Training ResNet WITH augmentation...")
    train_aug, val_aug, test_aug = get_dataloaders(task='B', augment=True, batch_size=64)
    _, hist_aug, acc_aug, y_true, y_pred, y_probs = train_resnet_with_tracking(
        train_aug, val_aug, test_aug, epochs=15, lr=0.001, use_augmentation=True
    )
    plot_training_history(hist_aug, "ResNet_With_Augmentation")
    results['ResNet_Aug'] = acc_aug

    logger.info(f"No Aug: {acc_no_aug:.4f} | With Aug: {acc_aug:.4f}")

    # Plot augmentation impact
    _plot_augmentation_impact(acc_no_aug, acc_aug)

    # Epochs Analysis
    logger.info("Experiment: Training Budget (Epochs)")
    analyze_epochs_impact(train_aug, val_aug, test_aug)

    return results, y_true, y_probs, y_pred


def _plot_augmentation_impact(acc_no_aug: float, acc_aug: float) -> None:
    """
    Helper function to visualize the impact of data augmentation.

    Args:
        acc_no_aug: Accuracy without augmentation.
        acc_aug: Accuracy with augmentation.
    """
    plt.figure(figsize=(7, 5))
    models = ['No Augmentation', 'With Augmentation']
    accs = [acc_no_aug, acc_aug]
    bars = plt.bar(models, accs, color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)

    plt.ylabel('Test Accuracy')
    plt.title('Impact of Data Augmentation on ResNet')
    plt.ylim([0.0, 1.0])  # Scale 0-1 for clarity

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'augmentation_comparison.png'), dpi=300)
    plt.close()


def perform_comparative_analysis(
    y_test_svm: np.ndarray,
    y_probs_svm: np.ndarray,
    y_pred_svm: np.ndarray,
    y_test_rn: np.ndarray,
    y_probs_rn: np.ndarray,
    y_pred_rn: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compares Model A and Model B performance using ROC, PR curves and medical metrics.

    Args:
        y_test_svm: Ground truth labels for SVM.
        y_probs_svm: Predicted probabilities for SVM.
        y_pred_svm: Predicted labels for SVM.
        y_test_rn: Ground truth labels for ResNet.
        y_probs_rn: Predicted probabilities for ResNet.
        y_pred_rn: Predicted labels for ResNet.

    Returns:
        Nested dictionary containing metrics for both models.
    """
    logger.info("="*40)
    logger.info("Starting Comparative Analysis")
    logger.info("="*40)

    # ROC Curves
    logger.info("Generating ROC Curve Comparison...")
    fpr_svm, tpr_svm, _ = roc_curve(y_test_svm, y_probs_svm)
    auc_svm = auc(fpr_svm, tpr_svm)

    fpr_rn, tpr_rn, _ = roc_curve(y_test_rn, y_probs_rn)
    auc_rn = auc(fpr_rn, tpr_rn)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svm, tpr_svm, linewidth=2, label=f'SVM+HOG (AUC={auc_svm:.3f})')
    plt.plot(fpr_rn, tpr_rn, linewidth=2, label=f'ResNet+Aug (AUC={auc_rn:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Classical vs Deep Learning')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'final_roc_comparison.png'), dpi=300)
    plt.close()

    # Precision-Recall Curves
    logger.info("Generating Precision-Recall Analysis...")
    ap_svm = plot_precision_recall_curve(y_test_svm, y_probs_svm, "SVM_HOG")
    ap_rn = plot_precision_recall_curve(y_test_rn, y_probs_rn, "ResNet_Aug")

    # Medical Metrics Analysis
    logger.info("Calculating Medical Metrics...")
    svm_metrics = analyze_medical_metrics(y_test_svm, y_pred_svm, "SVM_HOG")

    # Threshold probabilities for ResNet binary predictions if needed,
    # though y_pred_rn should already be binary from training function.
    resnet_metrics = analyze_medical_metrics(y_test_rn, y_pred_rn, "ResNet_Aug")

    return {
        'SVM': {**svm_metrics, 'auc': auc_svm, 'ap': ap_svm},
        'ResNet': {**resnet_metrics, 'auc': auc_rn, 'ap': ap_rn}
    }


def main() -> None:
    """
    Main orchestration function for the benchmarking study.
    """
    logger.info("="*80)
    logger.info("AMLS 25/26 - BENCHMARKING STUDY: BreastMNIST")
    logger.info("="*80)

    try:
        # Part 1: Model A Experiments
        svm_results, _, y_test_svm, y_probs_svm, y_pred_svm = run_svm_experiments()

        # Part 2: Model B Experiments
        rn_results, y_test_rn, y_probs_rn, y_pred_rn = run_resnet_experiments()

        # Part 3: Comparative Analysis
        metrics = perform_comparative_analysis(
            y_test_svm, y_probs_svm, y_pred_svm,
            y_test_rn, y_probs_rn, y_pred_rn
        )

        # Final Summary Log
        logger.info("="*80)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*80)

        logger.info("Model A (SVM):")
        logger.info(f"  Raw Accuracy:        {svm_results['SVM_Raw']:.4f}")
        logger.info(f"  HOG Accuracy:        {svm_results['SVM_HOG']:.4f}")
        logger.info(f"  AUC-ROC:             {metrics['SVM']['auc']:.4f}")
        logger.info(f"  Sensitivity:         {metrics['SVM']['sensitivity']:.4f}")

        logger.info("Model B (ResNet):")
        logger.info(f"  No Aug Accuracy:     {rn_results['ResNet_No_Aug']:.4f}")
        logger.info(f"  Aug Accuracy:        {rn_results['ResNet_Aug']:.4f}")
        logger.info(f"  AUC-ROC:             {metrics['ResNet']['auc']:.4f}")
        logger.info(f"  Sensitivity:         {metrics['ResNet']['sensitivity']:.4f}")

        best_model = "ResNet+Aug" if rn_results['ResNet_Aug'] > svm_results['SVM_HOG'] else "SVM+HOG"
        best_acc = max(rn_results['ResNet_Aug'], svm_results['SVM_HOG'])

        logger.info(f"Best Performing Model: {best_model} (Accuracy: {best_acc:.4f})")
        logger.info("="*80)
        logger.info(f"All experiments completed. Outputs saved to {FIGURES_DIR}")

    except Exception as e:
        logger.exception("An unexpected error occurred during execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
