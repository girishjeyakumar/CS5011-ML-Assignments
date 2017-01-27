from sklearn import linear_model
import matplotlib.pyplot as plt

from utilities import *

# Takes train and test data and returns
# modified one with unnecessary feature columns removed
# based on threshold and model coefficients
def feature_selection(X_train,X_test,coeff,thresh):
    new_X_train = []
    new_X_test = []
    for i in range(len(coeff)):
        if(abs(coeff[i])<thresh): continue
        new_X_train.append(X_train.transpose()[i])
        new_X_test.append(X_test.transpose()[i])
    return np.array(new_X_train).transpose(),np.array(new_X_test).transpose()

# Perform ridge regression on given data with the given alpha
def ridge_regression(alpha,X_train,y_train,X_test,y_test):

    rlr = linear_model.Ridge(alpha=alpha,fit_intercept=True, normalize=False)
    rlr.fit(X_train, y_train)

    rse = np.mean((rlr.predict(X_test) - y_test) ** 2)

    return rse,rlr.coef_[0],rlr.intercept_[0]


# Loading data from csv
train = load_data_csv("CandC-train" + str(1) + ".csv")
test = load_data_csv("CandC-test" + str(1) + ".csv")

# Getting the number of features
features = len(train[0]) - 1

# Splitting into features and column to be predicted
X_train, y_train = np.hsplit(train, [features])
X_test, y_test = np.hsplit(test, [features])

# Best fit model with alpha obtained from p6_1.py
rse,coeff,intercept = ridge_regression(1.5,X_train,y_train,X_test,y_test)

print "Intial number of features: %d\n"%(len(X_train[0]))
print "Intial RSE: %f\n"%(rse)
print "Coefficients: ", coeff
print "\nIntercept: %f\n" % (intercept)

thresh = 0.05

# Running model after feature selection
new_X_train, new_X_test = feature_selection(X_train,X_test,coeff,thresh)
rse,coeff,intercept = ridge_regression(1.5,new_X_train,y_train,new_X_test,y_test)

print "Number of features after feature selection: %d\n"%(new_X_train.shape[1])
print "RSE after feature selection: %f\n"%(rse)
print "Coefficients: ", coeff
print "\nIntercept: %f\n" % (intercept)