"""
Değerlendirme metrikleri ve görselleştirmeler.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PLOT_DIR


def compute_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }


def print_metrics(metrics, prefix=""):
    for key, val in metrics.items():
        print(f"{prefix}{key}: {val:.4f}")


def plot_training_curves(histories, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, h in enumerate(histories):
        axes[0].plot(h['train_loss'], alpha=0.7, label=f'Fold {i+1}')
        axes[1].plot(h['val_mae'], alpha=0.7, label=f'Fold {i+1}')
        axes[2].plot(h['lr'], alpha=0.7, label=f'Fold {i+1}')

    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].set_title('Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, save=True):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 0].plot(lims, lims, 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Actual vs Predicted')

    # Residual vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].set_title('Residuals vs Predicted')

    # Residual histogram
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_title('Residual Distribution')

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')

    plt.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, 'residual_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
