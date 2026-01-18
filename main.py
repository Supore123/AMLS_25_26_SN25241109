print("--- Script Initialized ---") # Debug: proves file is loading

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.metrics import roc_curve, auc
from Code.data_utils import get_dataloaders
from Code.A.model_a import ModelA
from Code.B.model_b import ModelB
from Code.early_stopping import EarlyStopping
from skimage.feature import hog
from skimage import exposure

def run_svm_experiment():
    print("\n--- Running Model A (SVM) ---")
    X_tr, y_tr, X_val, y_val, X_test, y_test = get_dataloaders(task='A')
    
    # Ensure Model A is using the fix: SVC(probability=True)
    # Use best params from your GridSearch results here
    # (e.g. C=10, gamma=0.01 depending on your heatmap)
    model = ModelA(use_hog=True) 
    
    # Fit the model
    # Note: If ModelA.run() runs the whole pipeline, we might need to fit manually 
    # to access probabilities. Let's do a clean manual fit here for the Report.
    print("   -> Training SVM (this may take a moment)...")
    model.pipeline.fit(model._extract_features(X_tr), y_tr)
    
    # Extract test features
    X_test_feat = model._extract_features(X_test)
    
    # Get probabilities for Class 1 (Malignant)
    try:
        y_probs = model.pipeline.predict_proba(X_test_feat)[:, 1]
    except AttributeError:
        print("ERROR: Model A predict_proba failed. Did you add 'probability=True' to SVC in model_a.py?")
        return None, None, 0, None

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, y_test

def run_resnet_experiment():
    print("\n--- Running Model B (ResNet) with Early Stopping ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loaders with Augmentation
    train_loader, val_loader, test_loader = get_dataloaders(task='B', augment=True)
    
    model = ModelB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # INITIALIZE EARLY STOPPING
    # Ensure early_stopping.py has 'self.val_loss_min = np.inf' (lowercase)
    early_stopping = EarlyStopping(patience=5, verbose=True, path='Code/B/best_resnet.pt')
    
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
        
        avg_val_loss = np.average(val_losses)
        
        # Check Early Stopping
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print("   -> Early Stopping triggered!")
            break
            
    # Load best model
    try:
        model.load_state_dict(torch.load('Code/B/best_resnet.pt', weights_only=True))
    except TypeError:
        # Fallback for older torch versions
        model.load_state_dict(torch.load('Code/B/best_resnet.pt'))
    
    # Test Evaluation
    model.eval()
    y_true, y_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def main():
    print("--- Starting Experiments ---")
    
    # 1. Run Experiments
    fpr_a, tpr_a, auc_a, _ = run_svm_experiment()
    
    if fpr_a is None:
        print("Stopping due to SVM Error.")
        return

    fpr_b, tpr_b, auc_b = run_resnet_experiment()
    
    # 2. Plot Combined ROC Curve
    print("--- Generating Final Plot ---")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_a, tpr_a, label=f'Model A (SVM+HOG) (AUC = {auc_a:.2f})')
    plt.plot(fpr_b, tpr_b, label=f'Model B (ResNet+Stop) (AUC = {auc_b:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Classical vs Deep Learning')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('final_roc_comparison.png')
    print("\n[SUCCESS] Generated 'final_roc_comparison.png'")

def visualize_hog_sample(loader):
    # Get one image
    images, _ = next(iter(loader))
    img = images[0].squeeze().cpu().numpy() # Get first image, remove channel dim

    # Calculate HOG
    _, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4),
                       cells_per_block=(1, 1), visualize=True)

    # Rescale intensity for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input Image')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG Features')

    plt.savefig("hog_visualization.png")
    print("Generated hog_visualization.png")

if __name__ == "__main__":
    main()
