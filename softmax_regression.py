import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, regularization='l2', lambda_param=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters # Number of training iterations
        #our model items (weight and bias)

        self.w = None
        self.b = None
        self.regularization = regularization
        self.lp = lambda_param

    # model for training data
    def fit(self, X, y):
        n_samples, n_dimension = X.shape #calculates the number of samples and dimensions in the data `X`
        n_classes = len(np.unique(y))
        #initialized model's weights and bias as arrays of zeros
        self.w = np.zeros((n_classes, n_dimension))
        self.b = np.zeros((n_classes,))

        # print(f"Training Softmax Regression with {self.regularization} regularization on {num_samples} samples...")

        # Calculate the linear model and Softmax probabilities
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.w.T) + self.b

            exps = np.exp(linear_model - np.max(linear_model, axis=1, keepdims=True))
            softmax = exps / np.sum(exps, axis=1, keepdims=True)

            # Encode true class labels in one-hot format
            y_encoded = np.zeros((n_samples, n_classes))
            y_encoded[np.arange(n_samples), y] = 1

            # Calculate gradients of the loss with respect to w and b
            gradient_weights = -(1/n_samples) * np.dot((y_encoded - softmax).T, X)
            gradient_bias = -(1/n_samples) * np.sum(y_encoded - softmax, axis=0)

            # Next we r adding regularization to the gradient
            if self.regularization == 'l2':
                gradient_weights += 2 * self.lp * self.w
            elif self.regularization == 'l1':
                gradient_weights += self.lp * np.sign(self.w)

            # next updating our model parameters
            self.w -= self.lr * gradient_weights
            self.b -= self.lr * gradient_bias

        #     if (i+1) % 50 == 0:
        #         loss = -np.sum(y_encoded * np.log(softmax + 1e-15)) / num_samples
        #         print(f"Iteration {i+1}: Loss {loss:.4f}")
        #
        # print("Training complete.")

    # Used to make predictions on new data X after the Softmax regression model was trained
    def predict(self, X):
        linear_model = np.dot(X, self.w.T) + self.b
        exps = np.exp(linear_model - np.max(linear_model, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        predictions = np.argmax(softmax, axis=1)
        # print("Predictions made.")
        return predictions