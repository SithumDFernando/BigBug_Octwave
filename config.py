"""Central configuration for the fraud detection pipeline."""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submissions')

# Random seed
SEED = 42

# Cross-validation
N_FOLDS = 5

# Feature engineering thresholds (derived from EDA)
NIGHT_HOURS = [0, 1, 2, 3]
NIGHT_HOURS_BROAD = [0, 1, 2, 3, 4, 5]
DEVICE_TRUST_THRESHOLD = 35
DEVICE_TRUST_THRESHOLD_V2 = 40
VELOCITY_THRESHOLD = 5
HIGH_AMOUNT_THRESHOLD = 500
VERY_HIGH_AMOUNT_THRESHOLD = 800

# Model hyperparameters (best from sweep)
GB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.2,
    'random_state': SEED,
}

# Target column
TARGET = 'is_fraud'
ID_COL = 'transaction_id'
