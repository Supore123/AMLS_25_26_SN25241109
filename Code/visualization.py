# In Code/visualization.py (NEW FILE)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join("Code", "output", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_learning_curve(estimator, X, y, title: str, cv: int = 5) -> None:
    """
    Critical plot showing train/val performance vs dataset size.
    Essential for discussing overfitting and model capacity.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score', linewidth=2)
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation score', linewidth=2)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='g')
    
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Learning Curve: {title}', fontsize=13)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"learning_curve_{title.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved learning curve to %s", out_path)
    
    # Analyze for overfitting
    gap = train_mean[-1] - val_mean[-1]
    logger.info("Final train-val gap: %.3f", gap)
    if gap > 0.15:
        logger.warning("OVERFITTING detected (gap > 0.15)")
    elif gap > 0.08:
        logger.warning("Slight overfitting (gap > 0.08)")
    else:
        logger.info("Well-generalized model")
