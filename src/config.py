"""
Hyperparameter ve proje ayarları.
Tüm sabitler burada — tek yerden kontrol.
"""
import os
import torch

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
HIDDEN_LAYERS = [512, 64, 512, 512, 256]
DROPOUT_RATE = 0.0
USE_BATCHNORM = False
ACTIVATION = 'SiLU'

# Eğitim
BATCH_SIZE = 128
LEARNING_RATE = 0.001535811480527823
WEIGHT_DECAY = 4.468938237778042e-05
EPOCHS = 300
PATIENCE = 20
T_0 = 50        # CosineAnnealingWarmRestarts periyodu
T_MULT = 1      # Restart multiplier
ETA_MIN = 1e-6  # Minimum LR

# Loss
LOSS_FN = 'MSE'  # 'MSE' veya 'Huber'
HUBER_DELTA = 1.0

# W&B
WANDB_PROJECT = 'dnn-universal-approx'
WANDB_ENABLED = False

# Paths
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
SUBMISSION_PATH = os.path.join(BASE_DIR, 'outputs', 'submission.csv')
