"""
Veri yükleme, ön işleme ve PyTorch Dataset.
Kritik: Her fold'da imputer+scaler AYRI fit edilir (data leakage yok).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from src.config import (
    TRAIN_PATH, TEST_PATH, FEATURES_TO_USE, FEATURES_TO_LOG,
    TARGET_COL, IMPUTE_K, N_FOLDS, SEED, BATCH_SIZE
)


class RegressionDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def load_raw_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def get_fold_splits(n_samples):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(kf.split(range(n_samples)))


def preprocess_fold(train_df, test_df, train_idx, val_idx, config=None):
    """
    Tek bir fold için tüm preprocessing pipeline'ını çalıştır.
    Returns: train_dataset, val_dataset, test_X_scaled, imputer, scaler
    """
    cfg = config or {}

    X_all = train_df[FEATURES_TO_USE].copy()
    y_all = train_df[TARGET_COL].values
    X_test = test_df[FEATURES_TO_USE].copy()

    X_train_raw = X_all.iloc[train_idx].copy()
    X_val_raw = X_all.iloc[val_idx].copy()
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    # KNN Imputation (fit on train fold ONLY)
    imputer = KNNImputer(n_neighbors=cfg.get('impute_k', IMPUTE_K))
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=FEATURES_TO_USE)
    X_val_imp = pd.DataFrame(imputer.transform(X_val_raw), columns=FEATURES_TO_USE)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES_TO_USE)

    # log1p dönüşümü
    for col in FEATURES_TO_LOG:
        X_train_imp[col] = np.log1p(X_train_imp[col].clip(lower=0))
        X_val_imp[col] = np.log1p(X_val_imp[col].clip(lower=0))
        X_test_imp[col] = np.log1p(X_test_imp[col].clip(lower=0))

    # StandardScaler (fit on train fold ONLY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    train_dataset = RegressionDataset(X_train_scaled, y_train)
    val_dataset = RegressionDataset(X_val_scaled, y_val)

    return train_dataset, val_dataset, X_test_scaled, imputer, scaler


def create_dataloaders(train_dataset, val_dataset, batch_size=BATCH_SIZE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
