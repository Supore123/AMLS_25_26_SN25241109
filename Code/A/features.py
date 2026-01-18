import numpy as np

def flatten_images(dataset):
    """
    Converts dataset images into flat vectors for classical ML models.
    Input: PyTorch dataset
    Output: X (N, 784), y (N,)
    """
    X = []
    y = []

    for image, label in dataset:
        X.append(image.numpy().reshape(-1))  # Flatten 28x28 → 784
        y.append(label.item())

    return np.array(X), np.array(y)

