"""
Technical validation tests for ModelTrainer class.

These tests validate model initialization and training for Linear (Logistic),
XGBoost, and LightGBM architectures.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix

from src.pipeline.model_trainer import ModelTrainer
from src.utils.config import MODEL_TYPES, RANDOM_STATE


class TestModelTrainer:
    """Test suite for ModelTrainer technical validation."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing dummy classification data."""
        # 20 samples, 10 features
        X = csr_matrix(np.random.rand(20, 10))
        y = pd.Series([0, 1] * 10)
        return X, y

    def test_init_model_linear(self):
        """Test creation of logistic regression model."""
        trainer = ModelTrainer()

        # Le type 'linear' initialise maintenant une LogisticRegression
        model = trainer.init_model('linear')

        assert isinstance(model, LogisticRegression)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_init_model_xgboost(self):
        """Test creation of XGBoost model."""
        trainer = ModelTrainer()
        model = trainer.init_model('xgboost')

        assert isinstance(model, XGBClassifier)
        assert model.random_state == RANDOM_STATE

    def test_init_model_lightgbm(self):
        """Test creation of LightGBM model."""
        trainer = ModelTrainer()
        model = trainer.init_model('lightgbm')

        assert isinstance(model, LGBMClassifier)

    def test_init_model_invalid_type(self):
        """Test error handling for invalid model type."""
        trainer = ModelTrainer()

        with pytest.raises(ValueError, match="non supporté"):
            trainer.init_model('random_forest_inconnu')

    def test_train_model_logic(self, sample_data):
        """Test that the training process completes and returns the model."""
        trainer = ModelTrainer()
        X, y = sample_data

        # Test avec Linear
        model = trainer.init_model('linear')
        trained_model = trainer.train_model(model, X, y)

        # Assertions
        assert hasattr(trained_model, 'classes_'), "Model should be trained (have classes_)"
        assert len(trained_model.classes_) == 2, "Should be binary classification"

        # Test des prédictions
        preds = trained_model.predict(X)
        assert len(preds) == 20
        assert np.all((preds == 0) | (preds == 1)), "Predictions must be 0 or 1"

    def test_train_all_model_types(self, sample_data):
        """Verify that all models defined in config can be initialized and trained."""
        trainer = ModelTrainer()
        X, y = sample_data

        for model_type in MODEL_TYPES:
            model = trainer.init_model(model_type)
            trained_model = trainer.train_model(model, X, y)

            assert trained_model is not None
            # Vérification des attributs spécifiques
            if model_type == 'linear':
                assert hasattr(trained_model, 'coef_')
            else:
                # XGB et LGBM utilisent feature_importances_
                assert hasattr(trained_model, 'feature_importances_')

    def test_consistency_random_state(self):
        """Verify that models use the RANDOM_STATE from config."""
        trainer = ModelTrainer()

        model_xgb = trainer.init_model('xgboost')
        model_lgbm = trainer.init_model('lightgbm')

        assert model_xgb.get_params()['random_state'] == RANDOM_STATE
        assert model_lgbm.get_params()['random_state'] == RANDOM_STATE


def run_model_trainer_tests():
    """Function to run all ModelTrainer tests programmatically."""
    import pytest
    result = pytest.main([__file__, "-v", "--tb=short"])
    if result == 0:
        print("✅ All ModelTrainer tests passed!")
        return True
    else:
        print("❌ Some ModelTrainer tests failed!")
        return False


if __name__ == "__main__":
    run_model_trainer_tests()