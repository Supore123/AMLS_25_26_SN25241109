"""SVM model utilities for classical experiments (Model A).

This module contains a lightweight wrapper around an SVM pipeline
including feature extraction (HOG), hyperparameter search and plotting
helpers. All functions and methods are typed and use logging instead of
printing to stdout.
"""

from typing import Tuple, List, Dict

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
from sklearn import svm
import os

# Module logger
logger = logging.getLogger(__name__)

# Output directory for figures
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

class ModelA:
    """Support Vector Machine wrapper with optional HOG feature extraction.

    The class encapsulates a scikit-learn Pipeline and provides convenience
    methods for extracting features, running a grid search for hyperparameters,
    evaluating on test data, and producing diagnostic plots.

    Args:
        use_hog (bool): If True, extract HOG features from images before
            feeding them to the classifier. If False, flatten raw pixels.
    """

    def __init__(self, use_hog: bool = True) -> None:
        self.use_hog = use_hog

        # Define the pipeline steps: scaler then SVM with probability outputs
        self.pipeline: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(random_state=42, probability=True)),
        ])

        # Grid-search parameter space for the 'svm' step
        self.param_grid: Dict[str, List] = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__kernel": ["rbf"],
            "svm__gamma": ["scale", 0.1, 0.01],
        }

    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features from images according to the configured pipeline.

        Args:
            images (np.ndarray): Array of images. Expected shapes:
                - (N, 28, 28) or (N, 784) or (28, 28) single image.

        Returns:
            np.ndarray: Feature matrix of shape (N, d).
        """
        # Ensure images are (N, H, W) for feature extraction
        if images.ndim == 2:
            images = images.reshape(-1, 28, 28)

        if self.use_hog:
            logger.info("Extracting HOG features for %d images", len(images))
            feature_list: List[np.ndarray] = []
            for img in images:
                fd = hog(
                    img,
                    orientations=8,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1),
                    visualize=False,
                )
                feature_list.append(fd)
            return np.array(feature_list)

        # Fallback: flatten raw pixel intensities
        return images.reshape(images.shape[0], -1)

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Pipeline, float]:
        """Execute full SVM training pipeline including grid search and plots.

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): Datasets.

        Returns:
            Tuple[Pipeline, float]: The fitted best estimator and the test
                accuracy on `X_test`.
        """
        logger.info("Running SVM pipeline; HOG=%s", self.use_hog)

        # 1. Feature extraction
        X_tr_feat = self._extract_features(X_train)
        X_test_feat = self._extract_features(X_test)

        # 2. Hyperparameter tuning using GridSearchCV
        logger.info("Starting GridSearchCV (5-fold) for SVM hyperparameters")
        search = GridSearchCV(self.pipeline, self.param_grid, cv=5, verbose=1, n_jobs=-1)
        search.fit(X_tr_feat, y_train)

        best_model = search.best_estimator_
        logger.info("Best params found: %s", search.best_params_)

        # 3. Final evaluation
        test_preds = best_model.predict(X_test_feat)
        test_acc = accuracy_score(y_test, test_preds)
        logger.info("Test accuracy: %.4f", test_acc)

        # 4. Diagnostic plots
        self._plot_confusion_matrix(y_test, test_preds, test_acc)
        if self.use_hog:
            self._plot_grid_results(search)

        return best_model, test_acc

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, acc: float) -> None:
        """Plot and save a confusion matrix heatmap for the SVM predictions.

        Args:
            y_true (np.ndarray): Ground-truth labels.
            y_pred (np.ndarray): Predicted labels.
            acc (float): Accuracy score to include in the title.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )
        plt.title(f"SVM Confusion Matrix (Acc: {acc:.2f})")
        suffix = "hog" if self.use_hog else "raw"
        out_path = os.path.join(FIGURES_DIR, f"svm_confusion_{suffix}.png")
        plt.savefig(out_path)
        plt.close()

    def _plot_grid_results(self, search: GridSearchCV) -> None:
        """Plot a heatmap of GridSearchCV mean test scores.

        Args:
            search (GridSearchCV): Fitted GridSearchCV instance.
        """
        results = search.cv_results_

        # Reshape scores into (len(C), len(gamma)) for heatmap plotting
        try:
            scores = results["mean_test_score"].reshape(
                len(self.param_grid["svm__C"]), len(self.param_grid["svm__gamma"])
            )
        except Exception:
            # Fallback: flatten into a 1-row heatmap if reshaping fails
            scores = np.atleast_2d(results["mean_test_score"])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            scores,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=self.param_grid["svm__gamma"],
            yticklabels=self.param_grid["svm__C"],
        )
        plt.xlabel("Gamma")
        plt.ylabel("C (Regularization)")
        plt.title("SVM Hyperparameter Search Accuracy")
        out_path = os.path.join(FIGURES_DIR, "svm_grid_search_heatmap.png")
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved SVM diagnostic plots to %s", FIGURES_DIR)


def plot_svm_concept(save_path: str = "svm_concept_diagram.png") -> None:
    """Create and save a simple SVM hyperplane concept diagram.

    This utility is purely illustrative and does not depend on dataset files.

    Args:
        save_path (str): File path to save the generated figure.
    """
    # Create toy data for the diagram
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    Y = [0] * 20 + [1] * 20
    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X, Y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # Plot hyperplane and margins
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    plt.figure(figsize=(6, 5))
    plt.plot(xx, yy, "k-", label="Hyperplane")
    plt.plot(xx, yy_down, "k--", label="Margin")
    plt.plot(xx, yy_up, "k--")
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.title("SVM Hyperplane & Margin Concept")
    plt.legend()
    # Ensure figures directory exists and save
    if not os.path.isabs(save_path):
        save_path = os.path.join(FIGURES_DIR, os.path.basename(save_path))
    plt.savefig(save_path)
    plt.close()
    logger.info("Generated %s", save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    plot_svm_concept()
