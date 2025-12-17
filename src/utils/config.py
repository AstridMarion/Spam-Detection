"""
Configuration for Spam Detection ML Pipeline.

This module contains all configuration constants used throughout
the pipeline. Students don't need to modify this file.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Literal  # <--- AJOUTEZ 'Literal' ICI

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# File names
EMAIL_FILE = "email_spam.csv"
SMS_FILE = "sms_spam.csv"

# Column names
TARGET_COL = "label"
MES_COL = "message"

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Missing data threshold (drop columns with more than X% missing)
MISSING_THRESHOLD = 0.7

# Cross-validation splits
N_SPLITS = 4

# Train test split
TRAIN_TEST_SPLIT_SIZE = 0.2

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Tokenization parameters
TOKEN_REGEX = r"(\S+)"
NB_FEATURES = 5000

# CountVectorizer parameters
CV_LOWERCASE: bool = True
CV_STOP_WORDS: Literal['english'] = 'english'

# Types of vectorizer
VECTORIZER_TYPES: List[str] = ["count", "tfidf"]

# Default parameters
VECTORIZER_PARAMS: Dict[str, Any] = {
    # TfidfVectorizer ou CountVectorizer
    "ngram_range": (1, 1),
    "max_features": NB_FEATURES,
    "min_df": 1,
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random state for reproducibility
RANDOM_STATE = 3

# Model parameter
NB_ITERATIONS = 1000

LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': NB_ITERATIONS,
    'random_state': RANDOM_STATE
}

# Available model types
MODEL_TYPES = ["linear", "xgboost", "lightgbm"]

# Default hyperparameter grids for optimization
DEFAULT_PARAM_GRIDS = {
    "linear": {
        'C': [0.1, 1.0, 10.0],
    },
    "xgboost": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    },
    "lightgbm": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

}

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================)

MLFLOW_EXPERIMENT_NAME = "spam_detection_ml"
MLFLOW_TRACKING_URI = "./mlruns"

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Metrics to calculate for binary classification
METRICS = ["accuracy", "precision", "recall"]

def get_data_file_path(filename):
    """Get full path to a data file."""
    return DATA_PATH / filename
