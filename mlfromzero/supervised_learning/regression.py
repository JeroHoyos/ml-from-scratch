import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
    
    # start with _ to indicate that is a private method
    def _initialize_params(self, X):
        n_features = X.shape[1]
        
        # Uniform Glorot Initialization
        limit = np.sqrt(6 / (n_features + 1))
        self.w = np.random.uniform(-limit, limit, (n_features,))
        self.b = 0.0

    def fit(self, X, y, num_iters=1000):

        self._initialize_params(X)
        m = X.shape[0]

        for _ in range(num_iters):

            y_pred = np.dot(X, self.w) + self.b

            dw = X.T.dot(y_pred - y) / m
            db = np.sum(y_pred - y) / m

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    def predict(self, X):


        return np.dot(X, self.w) + self.b