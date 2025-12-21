import numpy as np
from sklearn.svm import SVC
from .preprocess import apply_hog  # Assuming you write this in preprocess.py

def run_experiment(train_ds, val_ds, test_ds):
    # Convert PyTorch-style dataset to NumPy for sklearn
    X_train = train_ds.imgs
    y_train = train_ds.labels.flatten()
    X_test = test_ds.imgs
    y_test = test_ds.labels.flatten()

    # Task requirement: Compare Raw vs Processed
    # 1. Raw
    X_train_raw = X_train.reshape(len(X_train), -1)
    # 2. Processed (e.g., HOG)
    X_train_hog = apply_hog(X_train) 
    
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X_train_raw, y_train)
    print(f"SVM Accuracy: {model.score(X_test.reshape(len(X_test), -1), y_test)}")
