"""Base classes for estimators, regressors, classifiers, and transformers."""


class Estimator:
    """Base class for all estimators."""
    
    def fit(self, X, y=None):
        """Fit the estimator to the data."""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError


class Regressor(Estimator):
    """Base class for regression models."""
    pass


class Classifier(Estimator):
    """Base class for classification models."""
    pass


class Transformer(Estimator):
    """Base class for transformers."""
    
    def transform(self, X):
        """Transform the data."""
        raise NotImplementedError
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)
