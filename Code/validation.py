from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def cross_validate_svm(X, y, model, cv: int = 5):
    """Perform k-fold cross-validation and save a results bar chart.

    Args:
        X: Feature matrix.
        y: Labels.
        model: scikit-learn estimator.
        cv (int): Number of folds.

    Returns:
        np.ndarray: Cross-validation scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    logger.info("Cross-Validation Scores: %s", scores)
    logger.info("Mean Accuracy: %.4f ± %.4f", scores.mean(), scores.std())

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, cv + 1), scores)
    plt.axhline(y=scores.mean(), color="r", linestyle="--", label=f"Mean: {scores.mean():.4f}")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Results (Showing Model Stability)")
    plt.legend()
    out_path = os.path.join(FIGURES_DIR, "cross_validation_results.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved cross-validation results to %s", out_path)
    return scores
