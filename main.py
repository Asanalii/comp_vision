import numpy as np
from load_data import load_data
from svm import SVM
from softmax_regression import SoftmaxRegression
from k_cross_validation import k_fold_cross_validation

#FOR SVM
# Load data
X_dataset_traing, Y_dataset_traing, X_dataset_test, Y_dataset_test, new_labels = load_data()

# Perform 5-fold cross validation
accuracies = k_fold_cross_validation(X_dataset_traing, Y_dataset_traing, k=5)

# # Print average validation accuracy (this is not necessary, for us just to see)
# average_accuracy = np.mean(accuracies)
# print(f"\nAverage Validation Accuracy: {average_accuracy * 100:.2f}%")

# Initialize SVM model, with parameters if we want, if not without them
# regularization is l1 or l2 IT'S IMPORTANT
modelSVM = SVM(learning_rate=0.1, n_iters=1000, regularization='l2')

# And now here fit the model with our dataset, then make prediction
modelSVM.fit(X_dataset_traing, Y_dataset_traing)
test_predictionsSVM = modelSVM.predict(X_dataset_test)

#For see the accuracy
test_accuracySVM = np.mean(test_predictionsSVM == Y_dataset_test)
print(f"\nTest Accuracy(SVM): {test_accuracySVM * 100:.2f}%")

#--------------------------------------------
#BELOW SOFTMAX REGRESSION

# Initialize SVM model, with parameters if we want, if not without them
# regularization is l1 or l2 IT'S IMPORTANT
modelSoftmaxRegression = SoftmaxRegression(learning_rate=0.1, n_iters=1000, regularization='l2')

# And now here fit the model with our dataset, then make prediction

modelSoftmaxRegression.fit(X_dataset_traing, Y_dataset_traing)
test_predictionsSoftmaxRegression = modelSoftmaxRegression.predict(X_dataset_test)

#For see the accuracy

test_accuracySoftmaxRegression = np.mean(test_predictionsSoftmaxRegression == Y_dataset_test)
print(f"\nTest Accuracy(Softmax): {test_accuracySoftmaxRegression * 100:.2f}%")