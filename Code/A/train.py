"""Simple training helpers for classical models used in experiments.

Provides small functions to train logistic regression and SVM models. All
functions are typed and log key actions.
"""

from typing import Any
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def train_logistic_regression(X_train: Any, y_train: Any) -> LogisticRegression:
    """Train a logistic regression classifier.

    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.

    Returns:
        LogisticRegression: Fitted logistic regression model.
    """
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    logger.info("Trained LogisticRegression with %d samples", len(X_train))
    return model


def train_svm(X_train: Any, y_train: Any, kernel: str = "linear") -> SVC:
    """Train an SVM classifier with the specified kernel.

    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        kernel (str): Kernel type to pass to scikit-learn's SVC.

    Returns:
        SVC: Fitted SVM model.
    """
    model = SVC(kernel=kernel, probability=True)
    model.fit(X_train, y_train)
    logger.info("Trained SVM (kernel=%s) with %d samples", kernel, len(X_train))
    return model

