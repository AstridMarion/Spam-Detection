"""
Feature engineering module for Spam Detection ML Pipeline.

This module handles feature extraction and selection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from utils.config import TOKEN_REGEX
from utils.logger import get_logger, LogLevel

# Import MLflow
import mlflow

class FeatureEngineer:
    """
    Feature engineer for spam detection.

    Handles text vectorization with CountVectorizer or TfidfVectorizer.
    """

    def __init__(self,
                 stop_words: str,
                 lowercase: bool,
                 remove_punctuation: bool,
                 number_placeholder: bool,
                 vectorizer_type: str = 'count',
                 max_features: int = 5000
                 ):

        """Initialize the feature engineer."""
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.remove_punctuation = remove_punctuation
        self.number_placeholder = number_placeholder

        self.list_stop_words = None
        self.vectorizer = None

        if self.stop_words == 'nltk':
            try:
                from nltk.corpus import stopwords
                self.list_stop_words = stopwords.words('english')
            except (LookupError, OSError):
                # Si les stopwords ne sont pas disponibles, les télécharger
                nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                self.list_stop_words = stopwords.words('english')

    def tokenization(self, train_messages, test_messages):
        """
        Tokenization of the message text.

        Args:
            train_messages : pandas.core.series.Series
            test_messages : pandas.core.series.Series
        Returns:
            Tuple: tokenized train_messages and test_messages
        """
        logger = get_logger()
        logger.info("Tokenization ...")

        if self.vectorizer_type == 'count':
            # Call CountVectorizer
            vectorizer = CountVectorizer(
                max_features=self.max_features,
                token_pattern=TOKEN_REGEX,
                lowercase=self.lowercase,
                stop_words=self.list_stop_words
            )
        else:
            # Call TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                token_pattern=TOKEN_REGEX,
                lowercase=self.lowercase,
                stop_words=self.list_stop_words
            )

        vectorizer.fit(train_messages)

        self.vectorizer = vectorizer

        # Add mlflow
        if mlflow.active_run():
            # Log class name TfidfVectorizer or CountVectorizer
            vec_name = self.vectorizer.__class__.__name__
            mlflow.log_params({
                f'{vec_name}.vocabulary_size': len(self.vectorizer.vocabulary_),
                f'{vec_name}.token_pattern': TOKEN_REGEX,
                f'{vec_name}.stop_words': self.stop_words,
                f'{vec_name}.lowercase': self.lowercase,
            })

        # Logging
        logger.success("All features vectorized")

        return self.vectorizer.transform(train_messages), self.vectorizer.transform(test_messages)