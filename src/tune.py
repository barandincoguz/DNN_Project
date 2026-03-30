"""
Optuna ile hyperparameter optimization.
"""
import argparse
import numpy as np
from src.config import SEED, INPUT_DIM
from src.train import train_all_folds, set_seed

try:
    import optuna
except ImportError:
    optuna = None


WIDTH_OPTIONS = [32, 48, 64, 96, 128, 192, 256, 384, 512]
BASELINE_BEST_PARAMS = {
    'n_layers': 5,
    'dim_0': 512,
    'dim_1': 64,
    'dim_2': 512,
    'dim_3': 512,
    'dim_4': 256,
    'dropout': 0.0,
    'use_batchnorm': False,
    'activation': 'SiLU',
    'lr': 0.001535811480527823,
    'weight_decay': 4.468938237778042e-05,
    'batch_size': 128,
    'T_0': 50,
    'T_mult': 1,
    'eta_min': 1e-6,
    'loss_fn': 'MSE',
    'impute_k': 10,
    'patience': 20,
}


def require_optuna():
    if optuna is None:
        raise ImportError(
            "`src.tune` icin Optuna gerekli. Notebook'ta once `pip install optuna` calistir."
        )
    return optuna


def best_params_to_config(best_params, epochs=300, verbose=True, save_models=True):
    n_layers = best_params['n_layers']
    config = {
        'input_dim': INPUT_DIM,
        'hidden_layers': [best_params[f'dim_{i}'] for i in range(n_layers)],
        'dropout': best_params['dropout'],
        'use_batchnorm': best_params['use_batchnorm'],
        'activation': best_params['activation'],
        'lr': best_params['lr'],
        'weight_decay': best_params['weight_decay'],
        'batch_size': best_params['batch_size'],
        'T_0': best_params['T_0'],
        'T_mult': best_params.get('T_mult', 1),
        'eta_min': best_params.get('eta_min', 1e-6),
        'loss_fn': best_params.get('loss_fn', 'MSE'),
        'impute_k': best_params.get('impute_k', 10),
        'epochs': epochs,
        'patience': best_params.get('patience', 20),
        'verbose': verbose,
        'save_models': save_models,
    }
    if config['loss_fn'] == 'Huber' and 'huber_delta' in best_params:
        config['huber_delta'] = best_params['huber_delta']
    return config


BASELINE_BEST_CONFIG = best_params_to_config(BASELINE_BEST_PARAMS)


def objective(trial):
    optuna_module = require_optuna()
    n_layers = trial.suggest_int('n_layers', 2, 6)
    hidden_layers = [
        trial.suggest_categorical(f'dim_{i}', WIDTH_OPTIONS)
        for i in range(n_layers)
    ]

    loss_fn = trial.suggest_categorical('loss_fn', ['MSE', 'Huber'])
    config = {
        'input_dim': INPUT_DIM,
        'hidden_layers': hidden_layers,
        'dropout': trial.suggest_float('dropout', 0.0, 0.30, step=0.025),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['ReLU', 'SiLU', 'Mish', 'GELU']),
        'lr': trial.suggest_float('lr', 3e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'T_0': trial.suggest_categorical('T_0', [25, 40, 50, 75, 100]),
        'T_mult': trial.suggest_categorical('T_mult', [1, 2]),
        'eta_min': trial.suggest_float('eta_min', 1e-7, 1e-5, log=True),
        'loss_fn': loss_fn,
        'impute_k': trial.suggest_categorical('impute_k', [3, 5, 7, 10, 15]),
        'epochs': 300,
        'patience': trial.suggest_categorical('patience', [20, 30, 40, 50]),
        'verbose': False,
        'save_models': False,
        'wandb_enabled': True,
        'wandb_group': trial.study.study_name,
    }
    if loss_fn == 'Huber':
        config['huber_delta'] = trial.suggest_float('huber_delta', 0.5, 2.5)

    def report_fold_result(fold_idx, fold_maes):
        running_mae = float(np.mean(fold_maes))
        trial.report(running_mae, step=fold_idx)
        if trial.should_prune():
            raise optuna_module.TrialPruned()

    try:
        set_seed()
        fold_maes, _, _ = train_all_folds(config, fold_callback=report_fold_result)
        return float(np.mean(fold_maes))
    except optuna_module.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


def run_tuning(n_trials=100):
    optuna_module = require_optuna()
    sampler = optuna_module.samplers.TPESampler(
        seed=SEED,
        multivariate=True,
        group=True,
        n_startup_trials=min(15, max(5, n_trials // 4)),
    )
    pruner = optuna_module.pruners.MedianPruner(
        n_startup_trials=min(10, max(5, n_trials // 5)),
        n_warmup_steps=2,
    )
    study = optuna_module.create_study(
        direction='minimize',
        study_name='dnn_regression',
        sampler=sampler,
        pruner=pruner,
    )
    study.enqueue_trial(BASELINE_BEST_PARAMS)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    print(f"\nEN İYİ SONUÇ: MAE = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study


def parse_args():
    parser = argparse.ArgumentParser(description='Run Optuna tuning for the DNN regressor.')
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Optuna deneme sayısı',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_tuning(n_trials=args.trials)
