from sklearn.base import BaseEstimator, TransformerMixin


class RemoveFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        features_to_drop = [
            feature for feature in self.features if feature in X.columns
        ]
        if len(features_to_drop) > 0:
            X.drop(columns=features_to_drop, inplace=True)
        return X
