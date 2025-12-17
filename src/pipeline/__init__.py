"""
Spam Detection ML Pipeline Package

This package contains the core machine learning pipeline components:
- DataProcessor: Data loading and preprocessing
- FeatureEngineer: Feature extraction and selection
- ModelTrainer: Model training and comparison
- Evaluator: Core model evaluation

Usage:
    from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator
"""

__version__ = "1.0.0"

# Solution avec TYPE_CHECKING pour satisfaire l'IDE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Ces imports ne sont exécutés que par les outils d'analyse statique (IDE, mypy)
    # Pas d'importation circulaire au runtime !
    from .data_processor import DataProcessor
    from .feature_engineer import FeatureEngineer
    from .model_trainer import ModelTrainer
    from .evaluator import Evaluator

def __getattr__(name):
    """Import paresseux pour éviter les importations circulaires au runtime."""
    if name == 'DataProcessor':
        from .data_processor import DataProcessor
        return DataProcessor
    elif name == 'FeatureEngineer':
        from .feature_engineer import FeatureEngineer
        return FeatureEngineer
    elif name == 'ModelTrainer':
        from .model_trainer import ModelTrainer
        return ModelTrainer
    elif name == 'Evaluator':
        from .evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'DataProcessor',
    'FeatureEngineer',
    'ModelTrainer',
    'Evaluator'
]