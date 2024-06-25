import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CleanOutliersTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove outliers from a DataFrame using the IQR method.

    Attributes
    ----------
    columns : list
        List of columns to check for outliers. If None, all numeric columns are considered.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        """
        Learn the IQR for each column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        self.iqr_values_ = {}
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.iqr_values_[col] = (Q1, Q3, IQR)

        return self

    def transform(self, X, y=None):
        """
        Remove outliers from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : None
            Ignored.

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame with outliers removed.
        """
        X_transformed = X.copy()

        for col in self.columns:
            Q1, Q3, IQR = self.iqr_values_[col]
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_transformed = X_transformed[
                (X_transformed[col] >= lower_bound)
                & (X_transformed[col] <= upper_bound)
            ]

        return X_transformed
