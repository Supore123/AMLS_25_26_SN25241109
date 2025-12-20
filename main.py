# main.py
from Code.data_loader import get_dataloaders

def main():
    print("Loading BreastMNIST...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        augment=False
    )

    # Grab one batch
    images, labels = next(iter(train_loader))

    print("Sanity check:")
    print(f"Images shape: {images.shape}")   # expected: [8, 1, 28, 28]
    print(f"Labels shape: {labels.shape}")   # expected: [8, 1]
    print(f"First label value: {labels[0].item()}")

    print("Data loading works.")

if __name__ == "__main__":
    main()

