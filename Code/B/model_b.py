import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        # Load ResNet18
        self.net = resnet18(weights=None)
        # Modify for 1-channel input (Grayscale)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        # Modify for Binary Classification
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.net(x)

def run_hyperparameter_sweep(train_loader, val_loader, test_loader, epochs=15):
    """
    MSc Level: Runs the model with different Learning Rates to find the optimal configuration.
    """
    learning_rates = [0.01, 0.001, 0.0001]
    results = {}
    
    print(f"\n--- Model B (ResNet) | Hyperparameter Sweep (LRs: {learning_rates}) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for lr in learning_rates:
        print(f"   -> Testing Learning Rate: {lr}")
        model = ModelB().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Scheduler: Drop LR by factor of 0.1 every 7 epochs (Standard practice)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        
        val_acc_history = []
        
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validation
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.float().to(device)
                    outputs = model(images)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc_history.append(correct / total)
        
        results[lr] = {
            'model': model,
            'val_history': val_acc_history,
            'final_val_acc': val_acc_history[-1]
        }
        print(f"      Final Val Acc: {val_acc_history[-1]:.4f}")

    # Select Best Model
    best_lr = max(results, key=lambda x: results[x]['final_val_acc'])
    print(f"\n   -> BEST Learning Rate found: {best_lr}")
    best_model = results[best_lr]['model']
    
    # PLOT SWEEP RESULTS (Evidence for Report)
    plt.figure(figsize=(8, 5))
    for lr, data in results.items():
        plt.plot(data['val_history'], label=f'LR={lr}')
    plt.title("Hyperparameter Search: Learning Rate Impact")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("resnet_lr_sweep.png")
    plt.close()

    # FINAL TEST EVALUATION ON BEST MODEL
    best_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = best_model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Full Report
    print("\n--- Final Model B Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    plt.figure(figsize=(5,5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"ResNet Confusion Matrix (LR={best_lr})")
    plt.savefig("resnet_confusion.png")
    
    return accuracy_score(y_true, y_pred)
