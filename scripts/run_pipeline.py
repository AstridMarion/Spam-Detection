#!/usr/bin/env python3
"""
Simple Air Quality ML Pipeline with Inline MLflow Integration

This pipeline includes MLflow logging directly in the main workflow without
utility functions, making it easy for students to understand.
"""

import csv
import sys
import time
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import DataProcessor, FeatureEngineer, ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import MODEL_TYPES, DEFAULT_PARAM_GRIDS, DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, VECTORIZER_TYPES

from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings

def run_pipeline(args):
    """
    Run the complete air quality prediction pipeline with inline MLflow integration.
    """
    csv.field_size_limit(1048576)
    start_time = time.time()
    logger = get_logger()

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

    #Reduce the amount of noise shown by mlflow
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

    # Creating Experiment name (easier to separate this way)
    experiment_name = "Spam Detection"
    for data in args.train_datasets:
        experiment_name += f"_{data}"

    experiment_name += "_to"

    for data in args.test_datasets:
        experiment_name += f"_{data}"

    # Configuration MLflow simple
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Create descriptive run name
    if args.run_name == '':
        run_name = f"spam_{args.model}"

        if args.optimize:
            run_name += "_opti"

    else:
       run_name = args.run_name

    timestamp = datetime.datetime.now().strftime("%Hh%M")

    mlflow.start_run(run_name=f"{run_name}_{timestamp}")

    # Set tags for Dataset and Model columns in MLflow UI
    # PARAMS
    mlflow.log_param("model.type", args.model)
    mlflow.log_param("model.optimize", args.optimize)

    # TAGS
    mlflow.set_tag("dataset.path", DATA_PATH)
    mlflow.set_tag("mlflow.note.content",
               f"Pipeline with {args.model} model.")

    run = mlflow.active_run()
    logger.info("Run started:", run is not None)


    try:
        # Pipeline header with configuration
        logger.header("SPAM DETECTION ML PIPELINE")

        with logger.indent():
            logger.info(f"Model: {args.model}")
            logger.info(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")

############################################### DATA PIPELINE ##########################################################

        # Create the dataset selection
        train_selection = list(args.train_datasets)
        test_selection = list(args.test_datasets)

        with logger.indent():
            logger.data_info(f"Train data : {train_selection}")
            logger.data_info(f"Test data : {test_selection}")

        if mlflow.active_run():
            mlflow.log_params({
                "train.sms": "SMS" in train_selection,
                "train.email": "EMAIL" in train_selection,
                "test.sms": "SMS" in test_selection,
                "test.email": "EMAIL" in test_selection,
            })

        # Initialize components
        processor = DataProcessor(train_selection, test_selection)
        engineer = FeatureEngineer(
            stop_words=args.stop_words,
            lowercase=args.lowercase,
            remove_punctuation=args.remove_punctuation,
            number_placeholder=args.number_placeholder,
            vectorizer_type=args.vectorizer_type,
            max_features=args.vocabulary_size
        )
        trainer = ModelTrainer()

        # Step 1: Data Loading and Preprocessing
        logger.step("Data Loading and Preprocessing", 1)
        with logger.timer("Data loading and preprocessing"):
            train_msg, train_lab, test_msg, test_lab = processor.load_and_preprocess(
                drop_duplicates=True,
                balance= not args.optimize # Shouldn't balance if there is cross-validation
            )

        # Step 2: Feature Engineering
        logger.step("Text Preprocessing and Tokenization", 2)
        with logger.timer("Text Preprocessing and Tokenization"):
            train_msg, test_msg = engineer.tokenization(train_msg, test_msg)

############################################### MODEL PIPELINE #########################################################

            # Initialize evaluator
            evaluator = Evaluator()

            # Step 3: Model Pipeline
            logger.step("Model Pipeline", 3)

            # Instance the variable -> will only be set if optimized
            model_type = args.model
            best_params = None
            trained_model = None

            if model_type not in MODEL_TYPES:
                raise Exception(f"The input model '{model_type}' is not recognised")

            # Initialization of the model
            initial_model = trainer.init_model(model_type)

            if not args.optimize:
                # CASE 1 — no optimization
                with logger.timer("Model Training (no optimization)"):
                    trained_model = trainer.train_model(initial_model, train_msg, train_lab)

            else:
                # CAS 2 — optimization requested
                # Check if the model has a grid
                param_grid = DEFAULT_PARAM_GRIDS.get(model_type, {})

                # If no grid, fallback to training on default values
                if not param_grid:
                    logger.warning(
                        f"No parameter grid for '{model_type}'. Training will be performed on default values")

                    with logger.timer("Model Training (no optimization)"):
                        trained_model = trainer.train_model(initial_model, train_msg, train_lab)

                else:
                    # Grid exists -> perform optimization
                    with logger.timer("Hyperparameter optimization"):
                        # NOTE: hyperparameter_optimization_cv doit être dans Evaluator
                        trained_model, best_params, _ = evaluator.hyperparameter_optimization_cv(
                            initial_model,
                            param_grid,
                            train_msg,
                            train_lab
                        )

            # Step 4: Model Evaluation
            logger.step("Model Evaluation", 4)


            # Evaluation on the test set
            test_predictions = trained_model.predict(test_msg)
            # test_metrics = evaluator.calculate_metrics(test_lab, test_predictions)
            test_metrics = evaluator.calculate_metrics(test_lab, test_predictions)  # On utilise l'instance 'evaluator'

            # Evaluation of the training set for comparison
            train_predictions = trained_model.predict(train_msg)
            train_metrics = evaluator.calculate_metrics(train_lab, train_predictions)

            # Logging metrics
            logger.info(
                f"TEST Results: Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")

            # Log test metrics in MLflow
            if mlflow.active_run():
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


################################################# SAVE MODEL ###########################################################

        # Step 5: Results Summary
        logger.step("Results Summary", 5)

        end_time = time.time()
        execution_time = format_time_elapsed(start_time, end_time)

        summary = {
            'Model': str(trained_model.__class__.__name__),
            'Optimized': args.optimize,
            'Execution Time': execution_time,
            'training.metrics': train_metrics,
            'testing.metrics': test_metrics,
            'model.best_params': best_params or 'Not optimised'
        }

        logger.results_summary(summary)

        # Add MLflow final results logging
        if mlflow.active_run():
            # Log final summary metrics
            mlflow.log_metric("test.accuracy", summary["testing.metrics"]["accuracy"])
            mlflow.log_metric("test.precision", summary["testing.metrics"]["precision"])
            mlflow.log_metric("test.recall", summary["testing.metrics"]["recall"])
            mlflow.log_param("execution.time_seconds", int(end_time - start_time))

            # Log results summary as artifact
            mlflow.log_dict(summary, "results_summary.json")

            # Log Model
            mlflow.sklearn.log_model(
                trained_model,
                signature=infer_signature(train_msg, train_lab),
                input_example=train_msg[:5],
                registered_model_name= experiment_name,
                tags={
                    "trained.sms": "SMS" in processor.train_selection,
                    "trained.email": "EMAIL" in processor.train_selection
                }
            )

        logger.pipeline_complete(end_time - start_time)

        return summary

    finally:
        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()

def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Spam Detection ML Pipeline (Text Classification)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train-datasets', nargs='+', default=['SMS'], choices=['SMS', 'EMAIL'],
        help='Datasets to use for training (e.g., SMS EMAIL)'
    )
    parser.add_argument(
        '--test-datasets', nargs='+', default=['SMS'], choices=['SMS', 'EMAIL'],
        help='Datasets to use for testing (e.g., SMS EMAIL)'
    )

    parser.add_argument(
        '--model', type=str, default='linear',
        choices=MODEL_TYPES,
        help='Model type to train (e.g., linear, svc, rf, xgb, lgbm)'
    )
    parser.add_argument(
        '--optimize', action='store_true',
        help='Enable hyperparameter optimization using cross-validation (GridSearchCV)'
    )

    parser.add_argument(
        '--vectorizer-type', type=str, default='tfidf',
        choices=VECTORIZER_TYPES,
        help='Vectorizer type (count, tfidf, or hash)'
    )
    parser.add_argument(
        '--vocabulary-size', type=int, default=5000,
        help='Maximum number of features (tokens) to use for the vectorizer'
    )
    parser.add_argument(
        '--stop-words', type=str, default='english',
        choices=['english', 'none'],
        help='Stop words to use (language specific or none)'
    )

    # Text Preprocessing switches
    parser.add_argument('--lowercase', action='store_true', help='Convert text to lowercase')
    parser.add_argument('--remove-punctuation', action='store_true', help='Remove punctuation from text')
    parser.add_argument('--number-placeholder', action='store_true', help='Replace numbers with a placeholder token')

    parser.add_argument(
        '--run-name', type=str, default='',
        help='Optional custom MLflow run name'
    )

    parser.add_argument(
        '--mlflow', action='store_true',
        help='Enable MLflow tracking and model logging'
    )


    parser.add_argument(
        '--log-level', type=str, default=LogLevel.NORMAL.name.lower(),
        choices=[l.name.lower() for l in LogLevel],
        help='Logging level: silent, normal, verbose'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (deprecated, use --log-level verbose)'
    )

    return parser.parse_args()

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Configure logging level
        if args.verbose:
            # Support legacy --verbose flag
            log_level = LogLevel.VERBOSE
        else:
            log_level = log_level_from_string(args.log_level)

        set_log_level(log_level)

        # Run pipeline
        run_pipeline(args)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())