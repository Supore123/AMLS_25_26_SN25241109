import os
from Code.utils.dataset import BreastMNISTDataset
from Code.A.features import flatten_images
from Code.A.train import train_logistic_regression, train_svm
from Code.utils.metrics import evaluate_classification

# === Config ===
DATASET_DIR = os.path.join("Datasets", "BreastMNIST")
MODEL_TYPE = "logistic"  # options: "logistic" or "svm"

# === Load dataset ===
train_data = BreastMNISTDataset(DATASET_DIR, "train")
val_data   = BreastMNISTDataset(DATASET_DIR, "val")

# === Extract features ===
X_train, y_train = flatten_images(train_data)
X_val, y_val     = flatten_images(val_data)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# === Train model ===
if MODEL_TYPE == "logistic":
    model = train_logistic_regression(X_train, y_train)
elif MODEL_TYPE == "svm":
    model = train_svm(X_train, y_train, kernel="linear")
else:
    raise ValueError("Unknown MODEL_TYPE")

# === Evaluate ===
y_val_pred = model.predict(X_val)
metrics = evaluate_classification(y_val, y_val_pred)

print("\nValidation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

