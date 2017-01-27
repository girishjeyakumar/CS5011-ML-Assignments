from utilities import class_stats, load_data_csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

'''
    Here I use the SVC implementation of sklearn-libsvm which is built on top
    of the actual libsvm as it is well documented and more efficient.
'''

# Trials for different values of the C parameter in the SVM Model
C = [10 ** i for i in [-9, -6, -3, -1, 1]]
gamma = [10 ** i for i in [-9, -6, -3, -1, 1]]


def linear_kernel(X_train, y_train):
    print "\n-----Fitting linear kernel for SVM------\n"

    # Linear Kernel
    model = SVC(kernel="linear")

    # Trials for different value of the C parameter in the SVM Model
    C = [0.000000000001, 0.000000001, 0.000001]

    # Estimating best parameters based on 5-fold cross-validation
    parameters = {'C': C}
    model = GridSearchCV(model, parameters, cv=5)
    model.fit(X_train, y_train)

    print "Best fit parameters by varying C and evaluating performance by using 5-fold cross validation:"
    print "C: ", model.best_params_['C']

    return model


def poly_kernel(X_train, y_train):
    print "\n-----Fitting polynomial kernel for SVM------\n"

    # Polynomial Kernel
    model = SVC(kernel="poly")

    # Trials for different value of parameters in the SVM Model
    # gamma = [0.0001, 0.001, 0.01]
    degree = [3, 5, 7, 9, 11, 13, 15, 20]

    # Estimating best parameters based on 5-fold cross-validation
    parameters = {'C': C, 'degree': degree, 'gamma': gamma}
    model = GridSearchCV(model, parameters, cv=5)
    model.fit(X_train, y_train)

    print "Best fit parameters by varying C, degree & gamma and evaluating performance by using 5-fold cross validation:"
    print "C: ", model.best_params_['C']
    print "degree: ", model.best_params_['degree']
    print "gamma: ", model.best_params_['gamma']

    return model


def rbf_kernel(X_train, y_train):
    print "\n-----Fitting rbf kernel for SVM------\n"

    # Gaussian Kernel
    model = SVC(kernel="rbf")

    # Trials for different value of parameters in the SVM Model
    # gamma = [0.0001, 0.001, 0.01, 0.1, 1]

    # Estimating best parameters based on 5-fold cross-validation
    parameters = {'C': C, 'gamma': gamma}
    model = GridSearchCV(model, parameters, cv=5)
    model.fit(X_train, y_train)

    print "Best fit parameters by varying C & gamma and evaluating performance by using 5-fold cross validation:"
    print "C: ", model.best_params_['C']
    print "gamma: ", model.best_params_['gamma']

    return model


def sigmoid_kernel(X_train, y_train):
    print "\n-----Fitting sigmoid kernel for SVM------\n"

    # Sigmoid Kernel
    model = SVC(kernel="sigmoid")

    # Trials for different value of parameters in the SVM Model
    # gamma = [0.0001, 0.001, 0.01, 0.1, 1]

    # Estimating best parameters based on 5-fold cross-validation
    parameters = {'C': C, 'gamma': gamma}
    model = GridSearchCV(model, parameters, cv=5)
    model.fit(X_train, y_train)

    print "Best fit parameters by varying C & gamma and evaluating performance by using 5-fold cross validation:"
    print "C: ", model.best_params_['C']
    print "gamma: ", model.best_params_['gamma']

    return model


def print_performance_metrics(model, X_test, y_test):
    print "\n----Model performance metrics------\n"

    y_pred = model.predict(X_test)
    class_stats(y_pred, y_test, 4)


# Load train and test set from csv
train = load_data_csv("DS2-train.csv")
test = load_data_csv("DS2-test.csv")

# Getting the number of features
features = len(train[0]) - 1

# Splitting into features and class
X_train, y_train = np.hsplit(train, [features])
X_test, y_test = np.hsplit(test, [features])

X_train = X_train
X_test = X_test
y_train = y_train.ravel()
y_test = y_test.ravel()

# SVM Models with different kernels

# linear_model = linear_kernel(X_train,y_train)
# print_performance_metrics(linear_model,X_test,y_test)
#
# rbf_model = rbf_kernel(X_train, y_train)
# print_performance_metrics(rbf_model, X_test, y_test)
#
# poly_model = poly_kernel(X_train, y_train)
# print_performance_metrics(poly_model, X_test, y_test)

sigmoid_model = sigmoid_kernel(X_train, y_train)
print_performance_metrics(sigmoid_model, X_test, y_test)
