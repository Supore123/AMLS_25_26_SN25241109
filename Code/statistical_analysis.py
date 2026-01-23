from scipy import stats
import numpy as np

def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
    """
    Calculate confidence intervals using bootstrap resampling.
    Shows you understand statistical uncertainty.
    """
    scores = []
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        score = metric_fn(y_true_sample, y_pred_sample)
        scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    mean_score = np.mean(scores)
    
    print(f"\n📊 Bootstrap Confidence Interval ({confidence*100}%):")
    print(f"   Mean: {mean_score:.4f}")
    print(f"   95% CI: [{lower:.4f}, {upper:.4f}]")
    
    return mean_score, lower, upper

# Usage in main:
from sklearn.metrics import accuracy_score
mean_acc, lower, upper = bootstrap_confidence_interval(
    y_test, y_pred, accuracy_score, n_bootstrap=1000
)
