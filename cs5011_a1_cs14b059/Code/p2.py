from sklearn import linear_model
import numpy as np

from metrics import print_classification_metrics
from utilities import load_data_csv

# Load train and test set from csv
train = load_data_csv("DS1-train.csv")
test = load_data_csv("DS1-test.csv")

# Getting the number of features
features = len(train[0]) - 1

# Splitting into features and classification
X_train, y_train = np.hsplit(train, [features])
X_test, y_test = np.hsplit(test, [features])

# Training model
lr = linear_model.LinearRegression()

# Fitting model
lr.fit(X_train, np.ravel(y_train))

# Predicting using model
y_pred = lr.predict(X_test)

# Thresholding
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1

# Printing out results
print lr.coef_, lr.intercept_
print_classification_metrics(y_test,y_pred)
