"""
Shared fixtures for all test modules (NLP Context).

This module contains common fixtures used across multiple test files
to ensure consistency and reduce code duplication.
It simulates the data flow from raw text to vectorized features.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure these match your config
from src.utils.config import TARGET_COL, MES_COL

# Download NLTK stopwords if not present (for tests only)
import nltk
import os

# Set NLTK data path to include project directory
nltk_data_path = Path(__file__).parent / "nltk_data"
if str(nltk_data_path) not in nltk.data.path:
    nltk.data.path.insert(0, str(nltk_data_path))

# Download stopwords if needed
try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')  # Test if stopwords are available
except (LookupError, OSError):
    print("\n📥 Downloading NLTK stopwords for tests...")
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK stopwords downloaded successfully.\n")


@pytest.fixture
def base_sample_data():
    """Create base sample data (DataFrame) for testing all modules."""
    # Create realistic NLP sample data
    # Mix of Ham (0) and Spam (1) messages
    data = [
        {MES_COL: "Hello how are you doing today?", TARGET_COL: 0},
        {MES_COL: "Meeting confirmed for tomorrow morning", TARGET_COL: 0},
        {MES_COL: "Can you send me the files?", TARGET_COL: 0},
        {MES_COL: "Lunch at 12?", TARGET_COL: 0},
        {MES_COL: "Don't forget the deadline", TARGET_COL: 0},
        {MES_COL: "Congratulations you won a free prize!", TARGET_COL: 1},
        {MES_COL: "Click here to claim your money now", TARGET_COL: 1},
        {MES_COL: "Urgent action required for your account", TARGET_COL: 1},
        {MES_COL: "Free bitcoins available", TARGET_COL: 1},
        {MES_COL: "Limited time offer buy now", TARGET_COL: 1},
    ]

    # Create larger dataset by repeating
    data = data * 5  # 50 rows total

    df = pd.DataFrame(data)

    # Ensure target is correct type (Int64 or int)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


@pytest.fixture
def sample_train_test_data(base_sample_data):
    """Create raw train/test split (Pandas Series) from base data."""
    # Simple manual split for testing purposes (80/20)
    split_idx = int(len(base_sample_data) * 0.8)

    train_df = base_sample_data.iloc[:split_idx].copy()
    test_df = base_sample_data.iloc[split_idx:].copy()

    # Return separated Series as the pipeline expects
    return (
        train_df[MES_COL],
        train_df[TARGET_COL],
        test_df[MES_COL],
        test_df[TARGET_COL]
    )


@pytest.fixture
def sample_features_data(base_sample_data):
    """
    Create sample vectorized data.
    Simulates the output of a FeatureEngineer or TfidfVectorizer.
    """
    df = base_sample_data.copy()

    # Use a real lightweight vectorizer to generate realistic sparse features
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(df[MES_COL])

    # Return the matrix and the labels
    return X, df[TARGET_COL]


@pytest.fixture
def sample_X_y(sample_features_data):
    """
    Create feature matrix X and target y for model testing.
    Standard fixture format for sklearn model tests.
    """
    X, y = sample_features_data

    # X is already a sparse matrix or array from the previous fixture
    # y is the Series of labels

    return X, y


@pytest.fixture
def sample_predictions():
    """Create sample predictions for evaluation testing (Classification)."""
    np.random.seed(42)
    n_samples = 50

    # Generate realistic binary ground truth (0 or 1)
    y_true = np.random.randint(0, 2, n_samples)

    # Generate predictions (mostly correct but with some errors)
    # Flip ~10% of bits to simulate errors
    noise_mask = np.random.random(n_samples) < 0.1
    y_pred = y_true.copy()
    y_pred[noise_mask] = 1 - y_pred[noise_mask]

    return y_true, y_pred


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary path for model saving/loading tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir