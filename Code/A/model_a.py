import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from sklearn import svm

class ModelA:
    def __init__(self, use_hog=True):
        """
        Model A: Support Vector Machine (SVM) with Hyperparameter Optimization.
        """
        self.use_hog = use_hog
        # Define the pipeline steps
        # Define the pipeline steps
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ENABLE PROBABILITY HERE for ROC curves to work
            ('svm', SVC(random_state=42, probability=True))
        ])

        # DEFINING THE SEARCH SPACE (MSc Level Rigour)
        # We search for the best C (Regularization) and Gamma (Kernel coefficient)
        self.param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['rbf'],
            'svm__gamma': ['scale', 0.1, 0.01]
        }

    def _extract_features(self, images):
        if len(images.shape) == 2:
            images = images.reshape(-1, 28, 28)

        if self.use_hog:
            print(f"   -> Extracting HOG features for {len(images)} images...")
            feature_list = []
            for img in images:
                fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
                         cells_per_block=(1, 1), visualize=False)
                feature_list.append(fd)
            return np.array(feature_list)
        else:
            return images.reshape(images.shape[0], -1)

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test):
        print(f"\n--- Model A (SVM) | Optimizing Hyperparameters | HOG: {self.use_hog} ---")

        # 1. Feature Extraction
        X_tr_feat = self._extract_features(X_train)
        X_test_feat = self._extract_features(X_test)

        # 2. Hyperparameter Tuning (Grid Search)
        # 5-Fold Cross Validation to ensure robustness
        print("   -> Running Grid Search (5-Fold CV)...")
        search = GridSearchCV(self.pipeline, self.param_grid, cv=5, verbose=1, n_jobs=-1)
        search.fit(X_tr_feat, y_train)

        best_model = search.best_estimator_
        print(f"   -> Best Params Found: {search.best_params_}")

        # 3. Final Evaluation
        test_preds = best_model.predict(X_test_feat)
        test_acc = accuracy_score(y_test, test_preds)
        print(f"   -> Test Accuracy: {test_acc:.4f}")

        # 4. Generate MSc Level Plots
        self._plot_confusion_matrix(y_test, test_preds, test_acc)

        # Only plot heatmap if we have enough results and it's the HOG run (for clarity)
        if self.use_hog:
            self._plot_grid_results(search)

        return test_acc

    def _plot_confusion_matrix(self, y_true, y_pred, acc):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title(f"SVM Confusion Matrix (Acc: {acc:.2f})")
        suffix = "hog" if self.use_hog else "raw"
        plt.savefig(f"svm_confusion_{suffix}.png")
        plt.close()

    def _plot_grid_results(self, search):
        # Extract results for Heatmap
        results = search.cv_results_
        scores = results['mean_test_score'].reshape(len(self.param_grid['svm__C']),
                                                    len(self.param_grid['svm__gamma']))

        plt.figure(figsize=(6, 5))
        sns.heatmap(scores, annot=True, fmt=".3f", cmap="viridis",
                    xticklabels=self.param_grid['svm__gamma'],
                    yticklabels=self.param_grid['svm__C'])
        plt.xlabel("Gamma")
        plt.ylabel("C (Regularization)")
        plt.title("SVM Hyperparameter Search Accuracy")
        plt.savefig("svm_grid_search_heatmap.png")
        plt.close()
        print("   -> Plots saved: svm_confusion_hog.png, svm_grid_search_heatmap.png")


    def plot_svm_concept():
        # Create fake data just for the diagram
        X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
        Y = [0] * 20 + [1] * 20
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(X, Y)

        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # Plot the margins
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        plt.figure(figsize=(6, 5))
        plt.plot(xx, yy, 'k-', label="Hyperplane")
        plt.plot(xx, yy_down, 'k--', label="Margin")
        plt.plot(xx, yy_up, 'k--')
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
        plt.title("SVM Hyperplane & Margin Concept")
        plt.legend()
        plt.savefig("svm_concept_diagram.png")
        print("Generated svm_concept_diagram.png")

if __name__ == "__main__":
    plot_svm_concept()
