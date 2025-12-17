"""
Technical validation tests for Evaluator class (Classification).

Validates metrics calculation (Accuracy, Precision, Recall) and
cross-validation for the Spam Detection pipeline.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

from src.pipeline.evaluator import Evaluator
from src.utils.config import N_SPLITS


class TestEvaluator:
    """Test suite for Evaluator classification validation."""

    @pytest.fixture
    def sample_predictions(self):
        """Fixture providing binary classification labels."""
        y_true = pd.Series([0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        # 8 échantillons : 6 corrects, 2 erreurs
        # Precision: TP=3 / (TP+FP=3+1) = 0.75
        # Recall: TP=3 / (TP+FN=3+1) = 0.75
        # Accuracy: 6/8 = 0.75
        return y_true, y_pred

    def test_calculate_metrics_basic(self, sample_predictions):
        """Test basic metrics calculation for classification."""
        evaluator = Evaluator()
        y_true, y_pred = sample_predictions

        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_true, y_pred)

        # Assertions sur les clés
        assert isinstance(metrics, dict), "Should return dictionary"
        assert 'accuracy' in metrics, "Should include Accuracy"
        assert 'precision' in metrics, "Should include Precision"
        assert 'recall' in metrics, "Should include Recall"
        assert 'f1_score' not in metrics, "F1-Score should NOT be in metrics"

        # Check values
        assert metrics['accuracy'] == 0.75
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with 100% correct classification."""
        evaluator = Evaluator()

        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0

    def test_calculate_metrics_all_wrong(self):
        """Test metrics with 0% correct classification."""
        evaluator = Evaluator()

        y_true = pd.Series([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        metrics = evaluator.calculate_metrics(y_true, y_pred)

        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0

    def test_cross_validate_model_structure(self):
        """Test cross-validation output structure."""
        evaluator = Evaluator()

        # Création de données factices (X: 20 samples, 2 features)
        X = pd.DataFrame(np.random.rand(20, 2))
        y = pd.Series([0, 1] * 10)
        model = LogisticRegression()

        cv_results = evaluator.cross_validate_model(model, X, y)

        # Assertions sur la structure du dictionnaire CV
        expected_metrics = [
            'accuracy_mean', 'accuracy_std',
            'precision_mean', 'precision_std',
            'recall_mean', 'recall_std'
        ]

        for m in expected_metrics:
            assert m in cv_results, f"Metric {m} missing in CV results"
            assert isinstance(cv_results[m], float)

    def test_precision_division_by_zero(self):
        """Test robustness when no positive samples are predicted."""
        evaluator = Evaluator()

        y_true = pd.Series([1, 1, 1])
        y_pred = np.array([0, 0, 0])  # Le modèle ne prédit aucun spam

        # La précision devrait être 0.0 (et non une erreur de division par zéro)
        precision = evaluator.precision_score(y_true, y_pred)
        assert precision == 0.0

    def test_evaluator_initialization(self):
        """Test evaluator initialization and methods existence."""
        evaluator = Evaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'accuracy_score')
        assert hasattr(evaluator, 'calculate_metrics')
        assert hasattr(evaluator, 'hyperparameter_optimization_cv')


def run_evaluator_tests():
    """Run tests programmatically."""
    import pytest
    result = pytest.main([__file__, "-v"])
    return result == 0


if __name__ == "__main__":
    run_evaluator_tests()