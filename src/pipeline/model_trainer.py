"""
Model training module for Spam Detection ML Pipeline.

This module handles model training, comparison and evaluation.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import LOGISTIC_REGRESSION_PARAMS, RANDOM_STATE, MODEL_TYPES
from utils.logger import get_logger

# Import mlflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm


class ModelTrainer:
    """
    Model trainer class for Spam detection classification.

    Handles training of the Logistic Regression model and model persistence.
    """

    def __init__(self):
        """Initialize the model trainer."""
        self.logger = get_logger()
        self.best_params = None
        self.model = None

    def init_model(self, model_type: str = 'linear'):
        """Initialize model dans log its parameters with mlflow."""
        logger = get_logger()
        logger.info(f"Initialization of the model : {model_type}")

        if model_type == "linear":
            model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        elif model_type == "xgboost":
            model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
        elif model_type == "lightgbm":
            model = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
        else:
            raise ValueError(f"Modèle '{model_type}' non supporté.")

        # Logging parameters
        if mlflow.active_run():
            mlflow.log_param("model_family", model_type)
            mlflow.log_params(model.get_params())

        return model

    def train_model(self, model, X_train, y_train):
        """Train the mode and log it with mlflow."""
        model.fit(X_train, y_train)

        if mlflow.active_run():
            if isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(model, "model")
            elif isinstance(model, LGBMClassifier):
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

        return model


    def predict(self, model, X_train):
        """
        Make predictions using a trained model.

        Args:
            model: Trained model (classifier)
            X_train: Feature matrix

        Returns:
            Predictions array (0 or 1)
        """
        logger = get_logger()

        # Make predictions using model
        predictions = model.predict(X_train)

        # Logging
        logger.success(f"Generated {len(predictions)} predictions")

        return predictions