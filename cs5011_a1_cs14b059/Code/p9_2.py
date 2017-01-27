import os
import numpy as np
from utilities import class_stats,plot_points
from sklearn.metrics import confusion_matrix

def accuracy(y_pred,y_true,k):
    C = confusion_matrix(y_true,y_pred)
    correct=0
    for i in range(k): correct+=C[i][i]
    return float(correct)/np.sum(C)

# Trying out different values of lambdas to identifying the
# best fit

start = 0.0002

y_test = np.genfromtxt('Test_labels', skip_header=2)

max_acc = 0
best_lambd = start

# Collecting data for Accuracy vs Lambda plot
acc_val = []
lambd_val = []

for i in range(0,8):
    lambd = start*(2**i)

    # Running Boyds code for L1 regularized logistic regression
    os.system("l1_logreg_train -s Train_features Train_labels "+str(lambd)+" model")
    os.system("l1_logreg_classify model Test_features result")
    y_pred = np.genfromtxt('result', skip_header=7)

    # Collecting data for Accuracy vs Lambda plot
    acc = accuracy(y_test,y_pred,2)
    acc_val.append(acc)
    lambd_val.append(lambd)

    if max_acc < acc:
        max_acc = acc
        best_lambd = lambd

# Plotting Accuracy vs Lambda
plot_points(lambd_val,acc_val,"Lambda","Accuracy")

# Identifying best fit lambda and printing out results
print "Best fit lambda: ", best_lambd
os.system("l1_logreg_train -s Train_features Train_labels "+str(best_lambd)+" model")
os.system("l1_logreg_classify model Test_features result")
y_pred = np.genfromtxt('result', skip_header=7)
class_stats(y_pred,y_test,2)
