from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=self.contamination, random_state=42
        )

    def fit(self, X, y=None):
        self.isolation_forest.fit(X)
        return self

    def transform(self, X, y=None):
        is_inlier = self.isolation_forest.predict(X) == 1
        return X[is_inlier]
