"""
Data preprocessing utilities for Freight Cost Prediction.
Extracts vendor invoice records from SQLite and prepares features for regression modeling.
"""

import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd


def load_vendor_data(db_path: str) -> pd.DataFrame:
    """
    Load vendor invoice records from SQLite database.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file.

    Returns:
    --------
    pd.DataFrame: Raw records from the 'vendor_invoice' table.
    """
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM vendor_invoice"
        return pd.read_sql_query(query, conn)


def prepare_features(df: pd.DataFrame):
    """
    Prepare predictor feature (Dollars) and target variable (Freight).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing vendor invoice data.

    Returns:
    --------
    tuple: Features DataFrame (X) and target Series (y).
    """
    X = df[['Dollars']]
    y = df['Freight']
    return X, y


# Alias for backward compatibility
prepare_featres = prepare_features


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split feature matrix and target vector into train and test sets.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, default=0.2
        Fraction of dataset reserved for testing.
    random_state : int, default=42
        Seed for reproducible splitting.

    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
