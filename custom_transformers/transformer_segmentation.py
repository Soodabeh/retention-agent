import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class BehavioralSegmentation(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=4, features_to_use=None, random_state=42):
        self.n_clusters = n_clusters
        self.features_to_use = features_to_use
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')

    def fit(self, X, y=None):
        if self.features_to_use is None:
            self.features_to_use = X.columns.tolist()
        self.kmeans.fit(X[self.features_to_use])
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out['cluster_label'] = self.kmeans.predict(X[self.features_to_use])
        return X_out
