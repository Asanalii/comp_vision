import numpy as np
from svm import SVM
from softmax_regression import SoftmaxRegression


def k_fold_cross_validation(X, y, k=5):
    # Firstly create models with parameters ( SVM, softmax )

    modelSVM = SVM(learning_rate=0.1, n_iters=1000, regularization="l2")
    modelSoftmax = SoftmaxRegression(learning_rate=0.1, n_iters=1000, regularization="l2")

    # Then we r calculation size of each our fold
    fold_size = len(X) // k
    accuraciesSVM = []
    accuraciesSoftmax = []

    for i in range(k):
        # Create validation set
        start = i * fold_size
        end = (i + 1) * fold_size
        X_validation_set = X[start:end]
        y_validation_set = y[start:end]

        # Create training set
        X_training_set = np.concatenate((X[:start], X[end:]))
        y_training_set = np.concatenate((y[:start], y[end:]))

        # Train our SVM model
        modelSVM.fit(X_training_set, y_training_set)

        # then validate it
        predictionsSVM = modelSVM.predict(X_validation_set)
        accuracySVM = np.mean(predictionsSVM == y_validation_set)
        accuraciesSVM.append(accuracySVM)
        print(f"SVM - Fold {i + 1}: Accuracy: {accuracySVM * 100:.2f}%")

        # Train Softmax Regression model
        modelSoftmax.fit(X_training_set, y_training_set)

        # the validate it
        predictionsSoftmax = modelSoftmax.predict(X_validation_set)
        accuracySoftmax = np.mean(predictionsSoftmax == y_validation_set)
        accuraciesSoftmax.append(accuracySoftmax)
        print(f"Softmax Regression - Fold {i + 1}: Accuracy: {accuracySoftmax * 100:.2f}%")

    # Print average accuracies
    print(f"\n\nAverage SVM Accuracy: {np.mean(accuraciesSVM) * 100:.2f}%")
    print(f"Average Softmax Regression Accuracy: {np.mean(accuraciesSoftmax) * 100:.2f}%")

    return accuraciesSVM, accuraciesSoftmax