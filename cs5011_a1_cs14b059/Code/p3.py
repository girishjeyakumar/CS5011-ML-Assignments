from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from metrics import print_classification_metrics
from utilities import load_data_csv,plot_points

def perform_KNN(X_train,y_train,X_test,k):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train, np.ravel(y_train))
    y_pred = KNN.predict(X_test)
    acc = KNN.score(X_test,y_test)
    return y_pred,acc

# Load train and test set from csv
train = load_data_csv("DS1-train.csv")
test = load_data_csv("DS1-test.csv")

# Getting the number of features
features = len(train[0]) - 1

# Splitting into features and classification
X_train, y_train = np.hsplit(train, [features])
X_test, y_test = np.hsplit(test, [features])

# Collecting data for plot
k_val = []
acc_val = []

# k - neighbours considered
k=1

# Trying for different ks and recording error
for i in range(0,50):
    k += 5
    y_pred,acc = perform_KNN(X_train,y_train,X_test,k)
    k_val.append(k)
    acc_val.append(acc)

# Plotting
plot_points(k_val,acc_val,'K','Accuracy')

# Identifying the k with highest accuracy and printing out results
# for that model
l = np.argmax(acc_val)
print "Max accuracy of %f for K=%d"%(acc_val[l],k_val[l])
y_pred,acc = perform_KNN(X_train,y_train,X_test,k_val[l])
print_classification_metrics(y_test,y_pred)
