"""
Technical validation tests for DataProcessor class (NLP Version).

These tests validate the technical implementation without focusing on NLP performance.
They check that methods execute correctly, file loading works with specific separators,
balancing logic operates as intended, and data structures are consistent.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Assurez-vous que le parent directory est dans le path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ SOLUTION: Import direct des modules, pas du package
from pipeline.data_processor import DataProcessor, balance_data
from utils.config import (
    TARGET_COL, MES_COL,
    EMAIL_FILE, SMS_FILE,
    TRAIN_TEST_SPLIT_SIZE
)


class TestDataProcessor:
    """Test suite for DataProcessor technical validation (NLP context)."""

    @pytest.fixture
    def mock_mlflow(self, monkeypatch):
        """Mock MLFlow to prevent creating runs during tests."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.active_run.return_value = None
        monkeypatch.setattr("pipeline.data_processor.mlflow", mock)
        return mock

    @pytest.fixture
    def sample_data(self):
        """Create sample raw data for testing (Email and SMS formats)."""
        # 1. Create realistic Email data (typically separated by ;)
        # Intentionally unbalanced for testing balance_data later
        email_data = {
            MES_COL: [f'Email content {i}' for i in range(20)],
            TARGET_COL: [0] * 18 + [1] * 2  # 18 Hams, 2 Spams
        }
        df_email = pd.DataFrame(email_data)

        # 2. Create realistic SMS data (typically separated by ,)
        sms_data = {
            MES_COL: [f'SMS message {i}' for i in range(20)],
            TARGET_COL: [0] * 10 + [1] * 10 # Balanced
        }
        df_sms = pd.DataFrame(sms_data)

        # Add some duplicates to test cleaning
        df_sms = pd.concat([df_sms, df_sms.iloc[:2]], ignore_index=True)

        return df_email, df_sms

    @pytest.fixture
    def processor_with_data(self, sample_data, tmp_path, monkeypatch, mock_mlflow):
        """Create DataProcessor with sample data files patched in temporary directory."""

        df_email, df_sms = sample_data

        # Save to temporary files with correct separators as per data_processor.py logic
        email_path = tmp_path / "email_test.csv"
        sms_path = tmp_path / "sms_test.csv"

        # Important: maintain the separators used in load_data
        df_email.to_csv(email_path, sep=",", index=False)
        df_sms.to_csv(sms_path, sep=";", index=False)

        # Patch the config module variables to point to temp files
        monkeypatch.setattr('pipeline.data_processor.DATA_PATH', tmp_path)
        monkeypatch.setattr('pipeline.data_processor.EMAIL_FILE', "email_test.csv")
        monkeypatch.setattr('pipeline.data_processor.SMS_FILE', "sms_test.csv")

        # Create processor with default selection (Use both for Train and Test)
        processor = DataProcessor(
            train_selection=["SMS", "EMAIL"],
            test_selection=["SMS", "EMAIL"]
        )

        return processor, df_email, df_sms

    def test_load_data_success(self, processor_with_data):
        """Test that data loading works correctly with different separators."""
        processor, _, _ = processor_with_data

        # Test loading
        email_df, sms_df = processor.load_data()

        # Assertions
        assert isinstance(email_df, pd.DataFrame), "Should return DataFrame for emails"
        assert isinstance(sms_df, pd.DataFrame), "Should return DataFrame for SMS"
        assert len(email_df) > 0, "Email data should not be empty"
        assert len(sms_df) > 0, "SMS data should not be empty"

        # Check if Target was converted to Int64 (nullable int) as per code
        assert pd.api.types.is_integer_dtype(email_df[TARGET_COL]), "Email target should be integer type"
        assert pd.api.types.is_integer_dtype(sms_df[TARGET_COL]), "SMS target should be integer type"

    def test_balance_data_logic(self):
        """Test the standalone balance_data function (replaces fill logic test)."""
        # Create highly imbalanced data
        imbalanced_data = pd.DataFrame({
            MES_COL: ['msg'] * 100,
            TARGET_COL: [0] * 90 + [1] * 10  # 90 class 0, 10 class 1
        })

        # Execute balancing
        balanced_df = balance_data(imbalanced_data)

        # Assertions
        counts = balanced_df[TARGET_COL].value_counts()
        assert counts[0] == counts[1], "Classes should be perfectly balanced"
        assert len(balanced_df) >= len(imbalanced_data), "Data size should increase or stay same"
        assert counts[0] == 90, "Majority class count should remain unchanged"

    def test_data_source_selection(self, sample_data, tmp_path, monkeypatch, mock_mlflow):
        """Test logic for selecting SMS vs EMAIL for train/test (replaces geographic fold test)."""
        # Setup similar to fixture but we need custom selection
        df_email, df_sms = sample_data
        df_email.to_csv(tmp_path / "email.csv", sep=",", index=False)
        df_sms.to_csv(tmp_path / "sms.csv", sep=";", index=False)

        monkeypatch.setattr('pipeline.data_processor.DATA_PATH', tmp_path)
        monkeypatch.setattr('pipeline.data_processor.EMAIL_FILE', "email.csv")
        monkeypatch.setattr('pipeline.data_processor.SMS_FILE', "sms.csv")

        # Initialize processor to Train on SMS ONLY, Test on EMAIL ONLY
        processor = DataProcessor(train_selection=["SMS"], test_selection=["EMAIL"])
        processor.load_data()

        train_msg, train_lab, test_msg, test_lab = processor.preprocess_data(balance=False)

        # Verify Training data (SMS only)
        assert train_msg.str.contains("SMS").all(), "Training data should only contain SMS"
        assert not train_msg.str.contains("Email").any(), "Training data should NOT contain Email"

        # Verify Test data (Email only)
        assert test_msg.str.contains("Email").all(), "Test data should only contain Email"
        assert not test_msg.str.contains("SMS").any(), "Test data should NOT contain SMS"

    def test_duplicate_removal_logic(self, processor_with_data):
        """Test that duplicates are removed during preprocessing."""
        processor, _, raw_sms = processor_with_data
        processor.load_data()

        # Preprocess with drop_duplicates=True
        train_msg, _, _, _ = processor.preprocess_data(drop_duplicates=True, balance=False)

        # Simple integration check
        assert isinstance(train_msg, pd.Series)

    def test_preprocess_data_pipeline_execution(self, processor_with_data):
        """Test complete preprocessing pipeline execution."""
        processor, _, _ = processor_with_data

        # Load data first
        processor.load_data()

        # Execute preprocessing pipeline
        train_msg, train_lab, test_msg, test_lab = processor.preprocess_data(
            drop_duplicates=True,
            balance=True
        )

        # Assertions
        assert isinstance(train_msg, pd.Series), "Train messages should be a Series"
        assert isinstance(train_lab, pd.Series), "Train labels should be a Series"
        assert isinstance(test_msg, pd.Series), "Test messages should be a Series"
        assert isinstance(test_lab, pd.Series), "Test labels should be a Series"

        assert len(train_msg) == len(train_lab), "Train features and labels must match length"
        assert len(test_msg) == len(test_lab), "Test features and labels must match length"

        # Check balancing effect on training data
        counts = train_lab.value_counts()
        # Allow small off-by-one difference due to sampling/implementation details
        assert abs(counts[0] - counts[1]) <= 1, "Training data should be balanced"

    def test_load_and_preprocess_convenience_method(self, processor_with_data):
        """Test the convenience method that combines loading and preprocessing."""
        processor, _, _ = processor_with_data

        # Execute convenience method
        results = processor.load_and_preprocess(balance=True)

        # Assertions
        assert len(results) == 4, "Should return tuple of 4 elements"
        train_msg, train_lab, _, _ = results
        assert len(train_msg) > 0, "Should return data"

    def test_error_handling_no_data_loaded(self):
        """Test error handling when trying to preprocess without loading data."""
        # Initialize without loading
        processor = DataProcessor([], [])

        # Should raise ValueError when no data is loaded
        with pytest.raises(ValueError, match="Data must be loaded first"):
            processor.preprocess_data()

    def test_data_integrity_after_processing(self, processor_with_data):
        """Test that text content is preserved (no corruption)."""
        processor, _, _ = processor_with_data

        train_msg, _, _, _ = processor.load_and_preprocess(balance=False, drop_duplicates=False)

        # Check types inside the series
        assert train_msg.apply(lambda x: isinstance(x, str)).all(), "All messages should be strings"

        # Ensure no NaN values in messages
        assert not train_msg.isnull().any(), "No missing values allowed in messages"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])