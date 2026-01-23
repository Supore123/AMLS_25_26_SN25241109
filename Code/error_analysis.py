import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze_misclassifications(model, X_test, y_test, n_examples: int = 10):
    """Analyze which examples the model gets wrong and save visualizations.

    Args:
        model: Trained model with a .predict method.
        X_test: Test features.
        y_test: Test labels.
        n_examples (int): Number of misclassified examples to visualize.
    """
    y_pred = model.predict(X_test)

    # Find misclassified examples
    misclassified_idx = np.where(y_pred != y_test)[0]
    logger.info("Error analysis: %d misclassifications out of %d", len(misclassified_idx), len(y_test))

    # Analyze false positives vs false negatives
    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]

    logger.info("False Positives: %d, False Negatives: %d", len(fp_idx), len(fn_idx))

    # Visualize misclassified examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Misclassified Examples Analysis", fontsize=14)

    for i, idx in enumerate(misclassified_idx[:n_examples]):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", color="red")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "misclassification_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved misclassification analysis to %s", out_path)

    # Medical context
    if len(fn_idx) > len(fp_idx):
        logger.warning("More false negatives than false positives — missing malignant cases is critical")
    else:
        logger.info("More false positives than false negatives — less critical but increases unnecessary biopsies")
