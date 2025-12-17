"""
Spam Detection ML Package - Restructured

This package has been restructured into two main components:

1. Pipeline: Core machine learning pipeline components
   - DataProcessor: Data loading and preprocessing
   - FeatureEngineer: Feature extraction and selection
   - ModelTrainer: Model training and comparison
   - Evaluator: Core model evaluation

2. Utils: Utilities and configuration
   - config: Configuration constants and settings
   - utils: General utility functions
   - evaluation_utils: Detailed evaluation functions with visualizations

Usage:
    from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator
    from utils.config import *
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "2.0.0"
__author__ = "Spam Detection ML Workshop - Restructured"

# ✅ SUPPRIMÉ : Pas d'import ici pour éviter les imports circulaires
# Les utilisateurs doivent faire : from pipeline import DataProcessor
# Pas : from src import DataProcessor

__all__ = []  # Le package src/ ne devrait rien exporter directement