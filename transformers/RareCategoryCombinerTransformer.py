from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RareCategoryCombinerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency=0.01):
        self.min_frequency = min_frequency
        self.frequent_categories = {}

    def fit(self, X, y=None):
        for column in X.select_dtypes(include=["object", "category"]):
            freq = X[column].value_counts(normalize=True)
            self.frequent_categories[column] = freq[freq >= self.min_frequency].index
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column in X.select_dtypes(include=["object", "category"]):
            X[column] = np.where(
                X[column].isin(self.frequent_categories[column]), X[column], "Other"
            )
        return X

    def set_output(self, transform="default"):
        return self
