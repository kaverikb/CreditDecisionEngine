# config.py

import os

# Paths
DATA_RAW = 'data/raw'
DATA_PROCESSED = 'data/processed'
DATA_COLLECTIONS = 'data/collections'
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

# File paths
ACCEPTED_CSV = os.path.join(DATA_RAW, 'accepted_2007_to_2018q4.csv')
REJECTED_CSV = os.path.join(DATA_RAW, 'rejected_2007_to_2018q4.csv')

TRAIN_CSV = os.path.join(DATA_PROCESSED, 'train.csv')
VAL_CSV = os.path.join(DATA_PROCESSED, 'val.csv')
TEST_CSV = os.path.join(DATA_PROCESSED, 'test.csv')

DEFAULTED_CSV = os.path.join(DATA_COLLECTIONS, 'defaulted_borrowers.csv')

# Model paths
DEFAULT_MODEL = os.path.join(MODELS_DIR, 'default_model.pkl')
COLLECTIONS_MODEL = os.path.join(MODELS_DIR, 'collections_model.pkl')
ENCODERS = os.path.join(MODELS_DIR, 'encoders.pkl')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
MAX_DEPTH = 7
LEARNING_RATE = 0.05
N_ESTIMATORS = 100

# Risk thresholds for segmentation
RISK_LOW = 0.15
RISK_MEDIUM = 0.40

# Approval strategy thresholds
APPROVE_THRESHOLD_CONSERVATIVE = 0.25
APPROVE_THRESHOLD_MODERATE = 0.40
APPROVE_THRESHOLD_AGGRESSIVE = 0.60

# Business parameters
INTEREST_RATE_AVG = 0.12  # 12% average interest
LOSS_GIVEN_DEFAULT = 0.85  # 85% loss if default