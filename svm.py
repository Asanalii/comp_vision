import numpy as np

class SVM:
    #firstly identify the model with parameters (here we gave the common parameters, if u will decide not to write in main)
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000, regularization='l2'):
        self.lr = learning_rate
        self.lp = lambda_param
        self.n_iters = n_iters
        self.regularization = regularization
        # w and b are initialized as None.
        # These model parameters will be assigned during training.
        self.w = None
        self.b = None

    # next step to fit
    # used to train the SVM model on input data X and y
    def fit(self, X, y):
        n_samples, n_dimension = X.shape
        # converts the original class labels y into a binary format: ->
        # -> -1 for one class and 1 for the other class
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_dimension)
        self.b = 0

        # IN COMMENTS BELOW I VE WRITTEN JUST TO CHECK, IF YOU WANT U CAN CHECK ALSO

       #iterates over a fixed number of training iterations
        for i in range(self.n_iters):
            #(to test) total_loss = 0

            #iterates over each training sample in the dataset (X)
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    if self.regularization == 'l2':
                        #Here we r using for of regularization
                        # weight - learn_rate * (2 * lambda * weight)
                        self.w -= self.lr * (2 * self.lp * self.w)
                    elif self.regularization == 'l1':
                        # weight - learn_rate * ( lambda * sign(weight))
                        self.w -= self.lr * (self.lp * np.sign(self.w))
                    #(to test) total_loss += 0
                else:
                    if self.regularization == 'l2':
                        #Here again formula when the condition is false
                        self.w -= self.lr * (2 * self.lp * self.w - np.dot(x_i, y_[idx]))
                    elif self.regularization == 'l1':
                        self.w -= self.lr * (self.lp * np.sign(self.w) - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                    #(to test) total_loss += 1 - y_[idx] * (np.dot(x_i, self.w) - self.b)  # Hinge loss component

            #(to test) if (i + 1) % 50 == 0:  # Print updates every 50 iterations
            #(to test)     print(f"Iteration {i + 1}: Loss: {total_loss:.2f}")

    # make prediction about given data
    def predict(self, X):
        # calculates margin for X’ samples
        prediction = np.dot(X, self.w) - self.b
        return np.sign(prediction)