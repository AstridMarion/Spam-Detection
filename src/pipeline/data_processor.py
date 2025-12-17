"""
Data preprocessing module for Spam Detection NLP Pipeline.

This module handles data loading, cleaning, and preprocessing for
email and SMS spam classification tasks.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple

from utils.config import (
    DATA_PATH,  TRAIN_TEST_SPLIT_SIZE,
    TARGET_COL, EMAIL_FILE, SMS_FILE, RANDOM_STATE, MES_COL
)
from utils.logger import get_logger
import mlflow
from mlflow.data.pandas_dataset import from_pandas


def balance_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Balance training data by oversampling the minority class.

    Addresses class imbalance by randomly sampling additional instances
    from the underrepresented class until both classes have equal frequency.
    This prevents model bias toward the majority class.

    Parameters
    ----------
    data: pandas.Dataframe containing
        Training text messages
        Corresponding class labels (0 for ham, 1 for spam)

    Returns
    -------
    data:
        Dataframe containing the input data with equal
        class representation

    Notes
    -----
    Uses random sampling with replacement to increase minority class size.
    Preserves original data distribution while achieving balance.
    """
    logger = get_logger()

    # Working with a copy of the input data
    data = data.copy()

    counts = data[TARGET_COL].value_counts()

    logger.info("Label counts before balancing:\n" + str(counts))

    if counts[1] > counts[0]:
        label_to_oversample = 0
        diff = counts[1] - counts[0]
    else:
        label_to_oversample = 1
        diff = counts[0] - counts[1]

    draw_from = data[data[TARGET_COL] == label_to_oversample]

    # Fix: sample all at once with replace=True instead of loop with same random_state
    samples_to_add = draw_from.sample(n=diff, replace=True, random_state=RANDOM_STATE)
    data = pd.concat([data, samples_to_add], ignore_index=True)

    logger.info("Label counts after balancing:\n" + str(data[TARGET_COL].value_counts()))

    return data

class DataProcessor:
    """
    Data processor for spam detection datasets.

    Handles loading, cleaning, and preprocessing of email and SMS data
    for spam/ham classification.
    """

    def __init__(self, train_selection: List[str], test_selection: List[str]):
        """Initialize the data processor."""
        self.sms_data: Optional[pd.DataFrame] = None
        self.email_data: Optional[pd.DataFrame] = None
        self.train_selection: List[str] = train_selection
        self.test_selection: List[str] = test_selection
        self.logger = get_logger()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load email and SMS datasets from CSV files.

        Returns:
            Tuple of (email_df, sms_df)
        """
        self.logger.substep("Loading Data")

        email_path = DATA_PATH / EMAIL_FILE
        sms_path = DATA_PATH / SMS_FILE

        # EMAIL
        self.email_data = pd.read_csv(
            email_path,
            sep=","
            )

        # SMS
        self.sms_data = pd.read_csv(
            sms_path,
            sep=";"
            )

        # Typing so that mlflow won't warn about NaN not working with int64 (but Int64 does)
        self.sms_data[TARGET_COL] = self.sms_data[TARGET_COL].astype('Int64')
        self.email_data[TARGET_COL] = self.email_data[TARGET_COL].astype('Int64')

        with self.logger.indent():
            self.logger.dataframe_info(self.email_data, "email data")
            self.logger.dataframe_info(self.sms_data, "sms data")

        self.logger.success("Data loading completed")
        return self.email_data.copy(), self.sms_data.copy()


    def preprocess_data(self, drop_duplicates: bool = True, balance: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline.

        Args:
            drop_duplicates: Whether to drop duplicates before balancing
            balance: Whether to balance unequal classes

        Returns:
            Tuple of (train_msg, train_lab, test_msg, test_lab)
        """
        if self.sms_data is None or self.email_data is None:
            raise ValueError("Data must be loaded first. Call load_data()")

        logger = get_logger()
        logger.substep("Starting preprocessing pipeline...")

        #work with a copy
        sms_data = self.sms_data.copy()
        email_data = self.email_data.copy()

        # Step 1: drop duplicates
        if drop_duplicates:
            sms_data.drop_duplicates(inplace=True, ignore_index=True)
            email_data.drop_duplicates(inplace=True, ignore_index=True)

        # Step 2 : Create train/test df
        train_data = []
        test_data = []

        # SMS distribution
        if "SMS" in self.train_selection and "SMS" not in self.test_selection:
            train_data.append(sms_data)
        if "SMS" not in self.train_selection and "SMS" in self.test_selection:
            test_data.append(sms_data)
        if "SMS" in self.train_selection and "SMS" in self.test_selection:
            temp_test, temp_train = train_test_split(sms_data,
                                                     test_size=TRAIN_TEST_SPLIT_SIZE,
                                                     random_state=RANDOM_STATE,
                                                     stratify=sms_data[TARGET_COL])
            test_data.append(temp_test)
            train_data.append(temp_train)

        # Email distribution
        if "EMAIL" in self.train_selection and "EMAIL" not in self.test_selection:
            train_data.append(email_data)
        if "EMAIL" not in self.train_selection and "EMAIL" in self.test_selection:
            test_data.append(email_data)
        if "EMAIL" in self.train_selection and "EMAIL" in self.test_selection:
            temp_test, temp_train = train_test_split(email_data,
                                                     test_size=TRAIN_TEST_SPLIT_SIZE,
                                                     random_state=RANDOM_STATE,
                                                     stratify=email_data[TARGET_COL])
            test_data.append(temp_test)
            train_data.append(temp_train)

        # If only one df, it will simply be returned as is
        train_data = pd.concat(train_data, axis=0, ignore_index=True)
        test_data = pd.concat(test_data, axis=0, ignore_index=True)

        # Step 3: Balance the label proportions in train dataset
        if balance:
            train_data = balance_data(train_data)

        # Log data before splitting between message and label
        if mlflow.active_run():
            mlflow.log_input(
                from_pandas(train_data, DATA_PATH, targets=TARGET_COL),
                "Training",
                tags={
                    "contains.sms": str("SMS" in self.train_selection),
                    "contains.email": str("EMAIL" in self.train_selection)
                }
            )

            mlflow.log_input(
                from_pandas(test_data, DATA_PATH, targets=TARGET_COL),
                "Testing",
                tags={
                    "contains.sms": str("SMS" in self.test_selection),
                    "contains.email": str("EMAIL" in self.test_selection)
                }
            )

        # Step 4: Separate between message and label
        train_msg: pd.Series = train_data[MES_COL]
        train_lab: pd.Series = train_data[TARGET_COL]
        test_msg: pd.Series = test_data[MES_COL]
        test_lab: pd.Series = test_data[TARGET_COL]

        # Logging
        logger.success("Preprocessing pipeline completed")

        return train_msg, train_lab, test_msg, test_lab

    def load_and_preprocess(self, **preprocessing_kwargs) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Convenience method to load and preprocess data in one step.

        Args:
            **preprocessing_kwargs: Arguments for preprocess_data()

        Returns:
            Tuple of (train_msg, train_lab, test_msg, test_lab)
        """
        self.load_data()
        return self.preprocess_data(**preprocessing_kwargs)