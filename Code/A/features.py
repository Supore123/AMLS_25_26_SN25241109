"""Feature utilities for classical models (dataset conversion helpers)."""

from typing import Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def flatten_images(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a PyTorch dataset of images into flattened numpy arrays.

    Description:
        Iterates over a PyTorch dataset yielding (image, label) pairs and
        converts them into an (N, D) feature matrix and a label vector.

    Args:
        dataset: Iterable dataset returning (image, label). `image` is
            expected to be a torch.Tensor and `label` a scalar tensor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and label vector y.
    """
    X = []
    y = []

    for image, label in dataset:
        # Flatten 28x28 -> 784
        X.append(image.numpy().reshape(-1))
        y.append(label.item())

    X_arr = np.array(X)
    y_arr = np.array(y)
    logger.info("Flattened dataset to X shape=%s, y shape=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr

