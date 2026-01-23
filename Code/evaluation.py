import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_precision_recall_curve(y_true, y_probs, model_name: str) -> float:
    """Plot and save a precision-recall curve for a model.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities for the positive class.
        model_name (str): Identifier used for saved filename and legend.

    Returns:
        float: Average precision (area under PR curve).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    # Calculate F1 at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, "b-", linewidth=2.5, label=f"AP={avg_precision:.3f}")
    plt.scatter(
        recall[optimal_idx],
        precision[optimal_idx],
        c="red",
        s=150,
        zorder=5,
        edgecolors="black",
        linewidths=2,
        label=f"Optimal (F1={f1_scores[optimal_idx]:.3f})",
    )

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title(f"Precision-Recall Curve: {model_name}\n(Critical for Medical Diagnosis)", fontsize=13)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"precision_recall_{model_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Log detailed analysis (replaces prints)
    logger.info("%s - Precision-Recall Analysis: AP=%.4f", model_name, avg_precision)
    logger.info("Optimal decision threshold: %.4f (F1=%.4f)", optimal_threshold, f1_scores[optimal_idx])
    logger.info("Saved precision-recall figure to %s", out_path)

    # Medical interpretation logged at warning/info levels
    if recall[optimal_idx] < 0.80:
        logger.warning("Low recall (%.3f) — may miss malignant cases", recall[optimal_idx])
    else:
        logger.info("Good recall: %.1f%% of malignant cases detected", recall[optimal_idx] * 100)

    return avg_precision
