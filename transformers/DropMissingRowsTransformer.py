from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class DropMissingRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing_fraction=0.3):
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        # Convert to DataFrame if input is not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Calculate the fraction of missing values per row
        missing_fraction = X.isnull().mean(axis=1)

        # Filter rows where the fraction of missing values is less than or equal to max_missing_fraction
        return X[missing_fraction <= self.max_missing_fraction].reset_index(drop=True)
