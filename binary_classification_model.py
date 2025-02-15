import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class LogisticRegression_:
    def __init__(self, iterations: int, alpha: float, degree: int=1):
        self.iterations = iterations
        self.alpha = alpha
        self.degree = degree
        self.history = []


    def initialize_preprocessing(self):
        """ Initializes polynomial features and z-score standardization. """
        
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.scaler = StandardScaler()
        self.transformation = False

    
    def initialize_params(self, X: np.ndarray):
        """ Initializes the parameters for the model. """

        self.W = np.zeros((X.shape[1], 1))
        self.B = np.zeros((1, 1))
    

    def make_prediction(self, X: np.ndarray) -> np.ndarray:
        """ Makes prediction on given dataset. """

        if self.transformation == True:
            X = self.poly.transform(X)
            X = self.scaler.transform(X)

        Z = X @ self.W + self.B
        Y_hat = 1 / (1 + np.exp(-Z))

        return Y_hat


    def compute_cost(self, Y_hat: np.ndarray, Y: np.ndarray) -> np.float64:
        """ Computes the log loss cost. """

        cost = - np.mean(((Y * np.log(Y_hat + 1e-10)) + ((1 - Y) * np.log(1 - Y_hat + 1e-10))))

        return cost

    
    def compute_gradient(self, X: np.ndarray, Y_hat: np.ndarray, Y: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray]:
        """ Computes the gradients for the parameters update. """

        m = X.__len__()
        dW = (1 / m) * (X.T @ (Y_hat - Y))
        dB = (1 / m) * np.sum((Y_hat - Y), keepdims=True)
        
        return dW, dB
    

    def update_params(self, dW: np.ndarray, dB: np.ndarray):
        """ Updates the parameters having provided gradients. """

        self.W -= self.alpha * dW
        self.B -= self.alpha * dB
    

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """ Trains the binary classification model using gradient descent. """

        self.initialize_preprocessing()

        X_poly = self.poly.fit_transform(X)
        X_poly_scaled = self.scaler.fit_transform(X_poly)

        self.initialize_params(X_poly_scaled)

        for i in range(self.iterations):
            print(f"Iteration: {i+1}/{self.iterations}")

            Y_hat = self.make_prediction(X_poly_scaled)
            
            cost = self.compute_cost(Y_hat, Y)
            print(f"Cost: {cost}\n")

            dW, dB = self.compute_gradient(X_poly_scaled, Y_hat, Y)
            self.update_params(dW, dB)

            self.history.append((i+1, cost))

        self.transformation = True
