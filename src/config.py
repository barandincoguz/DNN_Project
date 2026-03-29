"""
Hyperparameter ve proje ayarları.
Tüm sabitler burada — tek yerden kontrol.
"""
import torch

# Reproducibility
SEED = 42
DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

# Data
FEATURES_TO_USE = ['f1', 'f2', 'f3', 'f4']  # f5 = tuzak, DROP
FEATURES_TO_LOG = ['f3']                       # log1p uygulanacak
TARGET_COL = 'target'
IMPUTE_K = 10
N_FOLDS = 5

# Model mimarisi
INPUT_DIM = 4  # len(FEATURES_TO_USE)
HIDDEN_LAYERS = [128, 256, 128, 64]
DROPOUT_RATE = 0.15
USE_BATCHNORM = True
ACTIVATION = 'SiLU'

# Eğitim
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 500
PATIENCE = 25
T_0 = 50        # CosineAnnealingWarmRestarts periyodu
T_MULT = 1      # Restart multiplier
ETA_MIN = 1e-6  # Minimum LR

# Loss
LOSS_FN = 'MSE'  # 'MSE' veya 'Huber'
HUBER_DELTA = 1.0

# W&B
WANDB_PROJECT = 'dnn-universal-approx'
WANDB_ENABLED = True

# Paths
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
MODEL_DIR = 'outputs/models'
PLOT_DIR = 'outputs/plots'
SUBMISSION_PATH = 'outputs/submission.csv'
