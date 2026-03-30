"""
Eğitim döngüsü: early stopping, LR scheduling, model kaydetme.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.config import (
    SEED, DEVICE, N_FOLDS, INPUT_DIM, HIDDEN_LAYERS, DROPOUT_RATE,
    USE_BATCHNORM, ACTIVATION, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    PATIENCE, T_0, T_MULT, ETA_MIN, LOSS_FN, HUBER_DELTA, MODEL_DIR, PLOT_DIR,
    WANDB_PROJECT, WANDB_ENABLED, IMPUTE_K
)
from src.model import DNN
from src.dataset import load_raw_data, get_fold_splits, preprocess_fold, create_dataloaders
from src.evaluate import compute_metrics

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loss_fn(config=None):
    cfg = config or {}
    loss_name = cfg.get('loss_fn', LOSS_FN)

    if loss_name == 'Huber':
        return nn.HuberLoss(delta=cfg.get('huber_delta', HUBER_DELTA))
    return nn.MSELoss()


def train_one_fold(fold, train_df, test_df, train_idx, val_idx, config=None):
    """Tek fold eğitimi."""
    cfg = config or {}
    verbose = cfg.get('verbose', True)

    if verbose:
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

    train_dataset, val_dataset, X_test_scaled, imputer, scaler = \
        preprocess_fold(train_df, test_df, train_idx, val_idx, config=cfg)
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=cfg.get('batch_size', BATCH_SIZE),
    )

    model = DNN(
        input_dim=cfg.get('input_dim', INPUT_DIM),
        hidden_layers=cfg.get('hidden_layers', HIDDEN_LAYERS),
        dropout=cfg.get('dropout', DROPOUT_RATE),
        use_batchnorm=cfg.get('use_batchnorm', USE_BATCHNORM),
        activation=cfg.get('activation', ACTIVATION),
    ).to(DEVICE)

    if verbose and fold == 0:
        print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', LEARNING_RATE),
        weight_decay=cfg.get('weight_decay', WEIGHT_DECAY)
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.get('T_0', T_0),
        T_mult=cfg.get('T_mult', T_MULT),
        eta_min=cfg.get('eta_min', ETA_MIN),
    )
    criterion = get_loss_fn(cfg)

    best_val_mae = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}

    for epoch in range(cfg.get('epochs', EPOCHS)):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets, val_losses = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        metrics = compute_metrics(val_targets, val_preds)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(metrics['MAE'])
        history['lr'].append(current_lr)

        # W&B logging
        wandb_enabled = cfg.get('wandb_enabled', WANDB_ENABLED)
        if wandb_enabled and wandb is not None:
            wandb.log({
                f'fold{fold}/train_loss': avg_train_loss,
                f'fold{fold}/val_loss': avg_val_loss,
                f'fold{fold}/val_mae': metrics['MAE'],
                f'fold{fold}/val_rmse': metrics['RMSE'],
                f'fold{fold}/val_r2': metrics['R2'],
                f'fold{fold}/lr': current_lr,
                'epoch': epoch,
            })

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val MAE: {metrics['MAE']:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Early stopping
        if metrics['MAE'] < best_val_mae:
            best_val_mae = metrics['MAE']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.get('patience', PATIENCE):
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    # Test predictions with best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        test_preds = model(X_test_tensor).cpu().numpy()

    if verbose:
        print(f"  Best Val MAE: {best_val_mae:.4f}")
    return best_val_mae, best_model_state, test_preds, history


def train_all_folds(config=None, fold_callback=None):
    """5-Fold CV tam eğitim."""
    set_seed()
    cfg = config or {}
    save_models = cfg.get('save_models', True)
    if save_models:
        os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    verbose = cfg.get('verbose', True)

    wandb_enabled = cfg.get('wandb_enabled', WANDB_ENABLED)

    # W&B init
    if wandb_enabled:
        if wandb is None:
            raise ImportError("wandb_enabled=True ama `wandb` kurulu degil.")
        wandb.init(
            project=WANDB_PROJECT,
            group=cfg.get('wandb_group'),
            config={
                'hidden_layers': cfg.get('hidden_layers', HIDDEN_LAYERS),
                'dropout': cfg.get('dropout', DROPOUT_RATE),
                'batch_size': cfg.get('batch_size', BATCH_SIZE),
                'lr': cfg.get('lr', LEARNING_RATE),
                'weight_decay': cfg.get('weight_decay', WEIGHT_DECAY),
                'epochs': cfg.get('epochs', EPOCHS),
                'patience': cfg.get('patience', PATIENCE),
                'T_0': cfg.get('T_0', T_0),
                'T_mult': cfg.get('T_mult', T_MULT),
                'eta_min': cfg.get('eta_min', ETA_MIN),
                'loss_fn': cfg.get('loss_fn', LOSS_FN),
                'activation': cfg.get('activation', ACTIVATION),
                'use_batchnorm': cfg.get('use_batchnorm', USE_BATCHNORM),
                'impute_k': cfg.get('impute_k', IMPUTE_K),
                'n_folds': N_FOLDS,
                'seed': SEED,
                'device': DEVICE,
            },
        )

    train_df, test_df = load_raw_data()
    fold_splits = get_fold_splits(len(train_df))

    fold_maes = []
    all_test_preds = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        mae, model_state, test_preds, history = \
            train_one_fold(fold, train_df, test_df, train_idx, val_idx, cfg)

        fold_maes.append(mae)
        all_test_preds.append(test_preds)
        all_histories.append(history)
        if save_models:
            torch.save(model_state, os.path.join(MODEL_DIR, f'model_fold{fold}.pt'))
        if fold_callback is not None:
            fold_callback(fold, fold_maes)

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    if verbose:
        print(f"\n{'='*50}")
        print(f"CV SONUÇ: MAE = {mean_mae:.4f} ± {std_mae:.4f}")
        print(f"Fold MAEs: {[f'{m:.4f}' for m in fold_maes]}")
        print(f"{'='*50}")

    # W&B summary
    if wandb_enabled and wandb is not None:
        wandb.log({
            'cv_mean_mae': mean_mae,
            'cv_std_mae': std_mae,
        })
        for i, mae in enumerate(fold_maes):
            wandb.log({f'fold{i}/best_mae': mae})
        wandb.finish()

    avg_test_preds = np.mean(all_test_preds, axis=0)
    return fold_maes, avg_test_preds, all_histories


if __name__ == '__main__':
    train_all_folds()
