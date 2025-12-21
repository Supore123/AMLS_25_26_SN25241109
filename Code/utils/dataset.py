import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BreastMNISTDataset(Dataset):
    def __init__(self, dataset_dir, split):
        assert split in ["train", "val", "test"]

        npz_path = os.path.join(dataset_dir, "breastmnist.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Expected breastmnist.npz in {dataset_dir}"
            )

        data = np.load(npz_path)

        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

