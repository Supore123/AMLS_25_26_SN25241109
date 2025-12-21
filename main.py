from Code.utils.dataset import BreastMNISTDataset

DATASET_DIR = "Datasets/BreastMNIST"

train_data = BreastMNISTDataset(DATASET_DIR, "train")
val_data   = BreastMNISTDataset(DATASET_DIR, "val")
test_data  = BreastMNISTDataset(DATASET_DIR, "test")

print(len(train_data), len(val_data), len(test_data))

