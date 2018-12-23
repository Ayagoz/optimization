import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class ExpTransformer(BaseEstimator, ClassifierMixin):

    def __init__(self, gamma=1.):
        self.gamma = gamma

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return 1 - np.exp(-self.gamma * X)

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def set_params(self, gamma):
        self.gamma = gamma

    def get_params(self, deep=False):
        return {'gamma': self.gamma}
