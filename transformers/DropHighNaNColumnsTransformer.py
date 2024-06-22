from sklearn.base import BaseEstimator, TransformerMixin


class DropHighNaNColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        missing_ratios = X.isnull().mean()
        self.columns_to_drop = missing_ratios[
            missing_ratios > self.threshold
        ].index.tolist()
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop)
