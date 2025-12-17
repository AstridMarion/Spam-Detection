"""
Spam Detection ML Pipeline - Test Suite

This package contains comprehensive tests for validating the technical implementation
of the Spam Detection ML Pipeline components.

Test Structure:
- test_data_processor.py: Technical validation for DataProcessor class
- test_feature_engineer.py: Technical validation for FeatureEngineer class (to be added)
- test_model_trainer.py: Technical validation for ModelTrainer class (to be added)
- test_integration.py: End-to-end integration tests (to be added)

Usage:
    # Run all tests
    pytest tests/

    # Run specific test module
    pytest tests/test_data_processor.py

    # Run with verbose output
    pytest tests/ -v

    # Run with coverage report
    pytest tests/ --cov=pipeline --cov=utils
"""

__version__ = "1.0.0"
__author__ = "Spam Detection ML Workshop - Test Suite"

# ✅ NE PAS importer les tests ici !
# Pytest les découvrira automatiquement
# Cela évite les imports circulaires

__all__ = []


def get_available_test_modules():
    """
    Get list of available test modules.

    Returns:
        dict: Dictionary mapping module names to their availability status
    """
    import os
    from pathlib import Path

    test_dir = Path(__file__).parent
    modules = {}

    # Détection automatique des fichiers de test
    for file in test_dir.glob("test_*.py"):
        module_name = file.stem.replace("test_", "")
        modules[module_name] = True

    return modules


def run_all_available_tests():
    """
    Run all available test modules using pytest.

    Returns:
        int: Exit code from pytest (0 = success)
    """
    import pytest
    from pathlib import Path

    test_dir = Path(__file__).parent

    # Run pytest programmatically
    exit_code = pytest.main([
        str(test_dir),
        "-v",
        "--tb=short"
    ])

    return exit_code


if __name__ == "__main__":
    # Permet de lancer tous les tests avec: python -m tests
    import sys
    sys.exit(run_all_available_tests())