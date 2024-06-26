from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return X[~((X < lower_bound) | (X > upper_bound)).any(axis=1)]

    def set_output(self, transform="default"):
        return self
