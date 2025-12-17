"""
Model evaluation module for Spam Detection ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Any, Dict
from sklearn.model_selection import KFold, GridSearchCV
from scipy.sparse import csr_matrix
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import N_SPLITS, RANDOM_STATE
from utils.logger import get_logger, LogLevel

# import mlflow
import mlflow

class Evaluator:
    """
    Core evaluator for Spam detection models.

    This class handles basic model evaluation including cross-validation
    and metrics calculation used by the package components.
    """

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def accuracy_score(self, truth: pd.Series, pred: np.ndarray) -> float :
        """Calculate accuracy as the proportion of correct predictions."""
        # Formula: (TP + TN) / (TP + TN + FP + FN)
        return (truth == pred).sum() / len(truth)

    def precision_score(self, truth: pd.Series, pred: np.ndarray, pos_label: int = 1) -> float:
        """Calculate precision for the positive class (spam)."""
        # Precision = TP / (TP + FP)
        t = np.array(truth)
        p = np.array(pred)
        tp = ((t == pos_label) & (p == pos_label)).sum()
        fp = ((t != pos_label) & (p == pos_label)).sum()

        if (tp + fp) == 0:
            return 0.0
        return tp / (tp + fp)

    def recall_score(self, truth: pd.Series, pred: np.ndarray, pos_label: int = 1) -> float:
        """Calculate recall for the positive class (spam)."""
        # Recall = TP / (TP + FN)
        t = np.array(truth)
        p = np.array(pred)
        tp = ((t == pos_label) & (p == pos_label)).sum()
        fn = ((t == pos_label) & (p != pos_label)).sum()

        if (tp + fn) == 0:
            return 0.0
        return tp / (tp + fn)

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """Calculate all key metrics using the custom score methods."""

        acc = self.accuracy_score(y_true, y_pred)
        prec = self.precision_score(y_true, y_pred)
        rec = self.recall_score(y_true, y_pred)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
        }

    def evaluate_model(self, model: Any, X_test: csr_matrix, testing_labels: pd.Series) -> dict:
        """Generate predictions and calculate metrics."""
        logger = get_logger()
        logger.info(f"Generating predictions with {model.__class__.__name__}")
        predictions = model.predict(X_test)
        metrics = self.calculate_metrics(testing_labels, predictions)

        if mlflow.active_run():
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        logger.info(
            f"Results: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        return metrics

    def cross_validate_model(self, model: Any, X: Any, y: Any) -> dict:
        """Perform cross-validation using KFold and log results."""
        logger = get_logger()
        logger.info(f"Cross-validating {model.__class__.__name__}...")
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_results = []

        # Conversion pour assurer la compatibilité avec .iloc
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y_df = pd.Series(y) if isinstance(y, np.ndarray) else y

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_df, y_df)):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            fold_metrics = self.calculate_metrics(y_val, y_pred)
            fold_results.append(fold_metrics)

            logger.info(f"Fold {fold + 1} - Accuracy: {fold_metrics['accuracy']:.4f}")

        # Agrégation
        cv_results = {}
        for metric in fold_results[0].keys():
            values = [f[metric] for f in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)

        if mlflow.active_run():
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_results.items()})

        print(f"  Average: Accuracy={cv_results['accuracy_mean']:.3f}±{cv_results['accuracy_std']:.3f}")
        return cv_results

    def hyperparameter_optimization_cv(self, model: Any, param_grid: dict, X: Any, y: Any):
        """Optimize hyperparameters using Accuracy as the scoring metric."""
        logger = get_logger()
        logger.info(f"Optimizing {model.__class__.__name__} using Accuracy...")

        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            n_jobs=-1
        )

        grid_search.fit(X, y)

        if mlflow.active_run():
            mlflow.log_params({f'best_{k}': v for k, v in grid_search.best_params_.items()})
            mlflow.log_metric('best_cv_accuracy', grid_search.best_score_)

        logger.info(f"Best Accuracy: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

