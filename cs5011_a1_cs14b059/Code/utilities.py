import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd


data_path = "./data/"

def save_data(filename, data):
    np.save(data_path + filename, data)

def load_data(filename):
    return np.load(data_path + filename)

def save_data_csv(filename,data):
    np.savetxt(data_path+filename,data,delimiter=",")

def load_data_csv(filename):
    return np.genfromtxt(data_path+filename,delimiter=",")

def random_split(data, f):
    train, test = train_test_split(data, train_size=f)
    return train, test

def class_stats(y_pred,y_true,k):
    C = confusion_matrix(y_true,y_pred)
    print C
    correct=0
    for i in range(k): correct+=C[i][i]

    acc = float(correct)/np.sum(C)
    print "Accuracy: ", acc

    for i in range(0,k):
       print "Class: ", i+1
       r = float(C[i][i])/sum(C[i])
       p = float(C[i][i])/(np.sum(C,axis=0)[i])
       f = 2*(p*r)/(p+r)
       print "Precision: ", p
       print "Recall: ", r
       print "F-measure: ", f,"\n---------"
    return acc,C


def plot_3d(X,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p=[]
    n=[]

    for i in range(len(X)):
        if(y[i]==0): p.append(list(X[i]))
        else: n.append(list(X[i]))

    p = np.array(p)
    n = np.array(n)

    ax.scatter(p[:,0],p[:,1],p[:,2], c='r', marker='o')
    ax.scatter(n[:,0], n[:,1], n[:,2], c='b', marker='o')

    plt.show()

def plot_2d_model(X,y,slope,intercept):

    p=[]
    n=[]

    for i in range(len(X)):
        if(y[i]==0): p.append(list(X[i]))
        else: n.append(list(X[i]))

    y_p = [0 for _ in range(0,len(p))]
    y_n = [0 for _ in range(0, len(n))]

    plt.scatter(p,y_p,c='r', marker='o')
    plt.scatter(n,y_n,c='b', marker='o')
    x_thresh = (0.5-intercept)/slope
    plt.plot([x_thresh,x_thresh],[1,-1])

    plt.show()

def plot_points(x,y,xlabel,ylabel):
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def plot_pr_curve(y_true,y_scores):

    for i in range(len(y_scores[0])):
        p, r, t = precision_recall_curve(y_true, y_scores[:,i], pos_label=1)
        plt.plot(r, p,label='Precision-recall curve of class {0}'''.format(i))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()