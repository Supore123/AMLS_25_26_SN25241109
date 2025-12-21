from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_logistic_regression(X_train, y_train):
    """
    Train a simple logistic regression classifier.
    """
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, kernel="linear"):
    """
    Train an SVM classifier with specified kernel (linear or rbf)
    """
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model

