"""
Technical validation tests for FeatureEngineer class.

These tests validate the feature engineering functionality including the
tokenization, vectorization (Count/TF-IDF), and text preprocessing.
"""

import pandas as pd
import numpy as np
import pytest
from scipy.sparse import issparse, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings

from src.pipeline.feature_engineer import FeatureEngineer
from src.utils.config import TOKEN_REGEX

# Suppress NLTK download messages during tests
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")


def _nltk_stopwords_available():
    """Check if NLTK stopwords are available."""
    try:
        from nltk.corpus import stopwords
        _ = stopwords.words('english')
        return True
    except (LookupError, OSError):
        return False


class TestFeatureEngineer:
    """Test suite for FeatureEngineer technical validation."""

    def test_initialization_count_vectorizer(self):
        """Test FeatureEngineer initialization with CountVectorizer."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False,
        )

        # Assertions
        assert engineer.lowercase is True, "Lowercase should be set to True"
        assert engineer.stop_words is None, "Stop words should be None"
        assert engineer.vectorizer_type == 'count', "Vectorizer type should be 'count'"
        assert engineer.max_features == 5000, "Number of features should be 5000"
        assert engineer.list_stop_words is None, "List of stop words should be None initially"

    def test_initialization_tfidf_vectorizer(self):
        """Test FeatureEngineer initialization with TfidfVectorizer."""
        engineer = FeatureEngineer(
            lowercase=False,
            stop_words=None,
            vectorizer_type='tfidf',
            max_features=3000,
            remove_punctuation=False,
            number_placeholder=False
        )

        # Assertions
        assert engineer.lowercase is False, "Lowercase should be set to False"
        assert engineer.vectorizer_type == 'tfidf', "Vectorizer type should be 'tfidf'"
        assert engineer.max_features == 3000, "Number of features should be 3000"

    def test_initialization_with_nltk_stopwords(self):
        """Test FeatureEngineer initialization with NLTK stopwords."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words='nltk',
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        # Assertions
        assert engineer.stop_words == 'nltk', "Stop words should be 'nltk'"
        assert engineer.list_stop_words is not None, "List of stop words should be loaded"
        assert isinstance(engineer.list_stop_words, list), "Stop words should be a list"
        assert len(engineer.list_stop_words) > 0, "Stop words list should not be empty"
        assert 'the' in engineer.list_stop_words, "Common stop word 'the' should be in list"

    def test_tokenization_count_vectorizer(self, sample_train_test_data):
        """Test tokenization with CountVectorizer."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages, _, test_messages, _ = sample_train_test_data

        # Tokenize
        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert issparse(train_features), "Train features should be sparse matrix"
        assert issparse(test_features), "Test features should be sparse matrix"
        assert train_features.shape[0] == len(train_messages), "Train features row count should match input"
        assert test_features.shape[0] == len(test_messages), "Test features row count should match input"
        assert train_features.shape[1] > 0, "Train features should have columns"
        assert test_features.shape[1] > 0, "Test features should have columns"
        assert train_features.shape[1] == test_features.shape[1], "Train and test should have same number of features"

    def test_tokenization_tfidf_vectorizer(self, sample_train_test_data):
        """Test tokenization with TfidfVectorizer."""
        engineer = FeatureEngineer(
            lowercase=False,
            stop_words=None,
            vectorizer_type='tfidf',
            max_features=3000,
            remove_punctuation=False,
            number_placeholder=False
        )
        train_messages, _, test_messages, _ = sample_train_test_data

        # Tokenize
        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert issparse(train_features), "Train features should be sparse matrix"
        assert issparse(test_features), "Test features should be sparse matrix"
        assert train_features.shape[0] == len(train_messages), "Train features row count should match input"
        assert test_features.shape[0] == len(test_messages), "Test features row count should match input"

        # TF-IDF values should be normalized
        # Check that row sums are approximately 1 (L2 normalization)
        row_norms = np.sqrt(np.array(train_features.multiply(train_features).sum(axis=1)).flatten())
        assert np.allclose(row_norms, 1.0, atol=1e-5), "TF-IDF vectors should be L2 normalized"

    def test_tokenization_with_stopwords(self, sample_train_test_data):
        """Test tokenization with stop words removal."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words='nltk',
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages, _, test_messages, _ = sample_train_test_data

        # Tokenize
        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert issparse(train_features), "Train features should be sparse matrix"
        assert issparse(test_features), "Test features should be sparse matrix"

        # With stop words removed, feature count should potentially be lower
        # (though not guaranteed depending on the corpus)
        assert train_features.shape[1] > 0, "Should still have features after stop word removal"

    def test_tokenization_lowercase_effect(self):
        """Test the effect of lowercase parameter."""
        messages_train = pd.Series(["HELLO World", "HELLO world"])
        messages_test = pd.Series(["hello WORLD"])

        # Without lowercase
        engineer_no_lower = FeatureEngineer(
            lowercase=False,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )
        train_no_lower, _ = engineer_no_lower.tokenization(messages_train, messages_test)

        # With lowercase
        engineer_lower = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )
        train_lower, _ = engineer_lower.tokenization(messages_train, messages_test)

        # With lowercase=True, "HELLO" and "hello" should be treated as the same token
        # So we should have fewer unique features
        assert train_lower.shape[1] <= train_no_lower.shape[1], \
            "Lowercase should reduce or maintain feature count"

    def test_tokenization_max_features_limit(self):
        """Test that max_features parameter limits vocabulary size."""
        # Create messages with many unique words
        train_messages = pd.Series([
            " ".join([f"word{i}" for i in range(100)])
        ])
        test_messages = pd.Series(["word1 word2"])

        max_features = 20
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=max_features,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert train_features.shape[1] <= max_features, \
            f"Feature count should not exceed max_features={max_features}"

    def test_tokenization_sparse_matrix_properties(self, sample_train_test_data):
        """Test sparse matrix properties of tokenization output."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages, _, test_messages, _ = sample_train_test_data
        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Check that matrices are in CSR format (efficient for row operations)
        assert isinstance(train_features, csr_matrix) or train_features.format == 'csr', \
            "Should return CSR sparse matrix"

        # Check sparsity (most values should be zero)
        train_density = train_features.nnz / (train_features.shape[0] * train_features.shape[1])
        assert train_density < 0.5, "Matrix should be sparse (density < 50%)"

    def test_tokenization_empty_messages(self):
        """Test tokenization with empty messages."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages = pd.Series(["hello world", "test message", ""])
        test_messages = pd.Series(["", "another test"])

        # Should not raise an error
        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert train_features.shape[0] == len(train_messages), "Should handle empty messages"
        assert test_features.shape[0] == len(test_messages), "Should handle empty messages"

    def test_tokenization_special_characters(self):
        """Test tokenization with special characters and punctuation."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages = pd.Series([
            "Hello!!! World???",
            "Test@123 #hashtag",
            "Price: $100.00"
        ])
        test_messages = pd.Series(["Test message!!!"])

        # Should handle special characters based on TOKEN_REGEX
        train_features, test_features = engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert train_features.shape[0] == len(train_messages), "Should process messages with special chars"
        assert train_features.shape[1] > 0, "Should extract some tokens"

    def test_tokenization_consistency(self, sample_train_test_data):
        """Test that tokenization is consistent across multiple calls."""
        train_messages, _, test_messages, _ = sample_train_test_data

        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        # First tokenization
        train1, test1 = engineer.tokenization(train_messages, test_messages)

        # Second tokenization (create new engineer)
        engineer2 = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )
        train2, test2 = engineer2.tokenization(train_messages, test_messages)

        # Assertions - shapes should be the same
        assert train1.shape == train2.shape, "Tokenization should be consistent"
        assert test1.shape == test2.shape, "Tokenization should be consistent"

    def test_count_vs_tfidf_difference(self, sample_train_test_data):
        """Test that Count and TF-IDF vectorizers produce different results."""
        train_messages, _, test_messages, _ = sample_train_test_data

        # Count vectorizer
        engineer_count = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )
        train_count, _ = engineer_count.tokenization(train_messages, test_messages)

        # TF-IDF vectorizer
        engineer_tfidf = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type = 'tfidf',
            max_features = 5000,
            remove_punctuation=False,
            number_placeholder=False
        )
        train_tfidf, _ = engineer_tfidf.tokenization(train_messages, test_messages)

        # Assertions
        assert train_count.shape == train_tfidf.shape, "Should have same shape"

        # Values should be different (Count uses raw counts, TF-IDF uses weighted scores)
        assert not np.allclose(train_count.toarray(), train_tfidf.toarray()), \
            "Count and TF-IDF should produce different values"

    def test_vectorizer_is_stored(self, sample_train_test_data):
        """Test that vectorizer is properly stored after tokenization."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=5000,
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages, _, test_messages, _ = sample_train_test_data

        # Tokenize
        engineer.tokenization(train_messages, test_messages)

        # Assertions
        assert hasattr(engineer, 'vectorizer'), "Vectorizer should be stored as attribute"
        assert engineer.vectorizer is not None, "Vectorizer should not be None"
        assert isinstance(engineer.vectorizer, (CountVectorizer, TfidfVectorizer)), \
            "Vectorizer should be CountVectorizer or TfidfVectorizer"
        assert hasattr(engineer.vectorizer, 'vocabulary_'), "Vectorizer should be fitted (have vocabulary_)"

    def test_vocabulary_size(self, sample_train_test_data):
        """Test that vocabulary size is reasonable."""
        engineer = FeatureEngineer(
            lowercase=True,
            stop_words=None,
            vectorizer_type='count',
            max_features=100,  # Limit to 100 features
            remove_punctuation=False,
            number_placeholder=False
        )

        train_messages, _, test_messages, _ = sample_train_test_data

        # Tokenize
        train_features, _ = engineer.tokenization(train_messages, test_messages)

        # Assertions
        vocab_size = len(engineer.vectorizer.vocabulary_)
        assert vocab_size <= 100, "Vocabulary size should not exceed max_features"
        assert vocab_size == train_features.shape[1], "Vocabulary size should match feature count"


def run_feature_engineer_tests():
    """
    Function to run all FeatureEngineer tests programmatically.
    """
    import pytest

    result = pytest.main([__file__, "-v", "--tb=short"])

    if result == 0:
        print("✅ All FeatureEngineer tests passed!")
        return True
    else:
        print("❌ Some FeatureEngineer tests failed!")
        return False


if __name__ == "__main__":
    run_feature_engineer_tests()