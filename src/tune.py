"""
Optuna ile hyperparameter optimization.
"""
import numpy as np
import optuna
from src.config import SEED, INPUT_DIM
from src.train import train_all_folds, set_seed


def objective(trial):
    n_layers = trial.suggest_int('n_layers', 3, 6)
    hidden_layers = [
        trial.suggest_categorical(f'dim_{i}', [64, 128, 256, 512])
        for i in range(n_layers)
    ]

    config = {
        'input_dim': INPUT_DIM,
        'hidden_layers': hidden_layers,
        'dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.05),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['ReLU', 'SiLU', 'Mish', 'GELU']),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'T_0': trial.suggest_categorical('T_0', [30, 50, 75]),
        'epochs': 300,
        'patience': 20,
    }

    try:
        set_seed()
        fold_maes, _, _ = train_all_folds(config)
        return np.mean(fold_maes)
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


def run_tuning(n_trials=100):
    study = optuna.create_study(
        direction='minimize',
        study_name='dnn_regression',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nEN İYİ SONUÇ: MAE = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study


if __name__ == '__main__':
    run_tuning(n_trials=100)
