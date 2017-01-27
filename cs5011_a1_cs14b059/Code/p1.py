import numpy as np
from utilities import save_data_csv,random_split

def rand_covar(n):
    A = np.random.rand(n, n)
    covar = np.dot(A, A.transpose())
    return covar

def rand_mean(n):
    return np.random.rand(n)

features = 20
n = 2000

# Generating random means
mean_0 = rand_mean(features)
mean_1 = mean_0+rand_mean(features)*(10**-1)

# Generating random covariance matrix
cov = rand_covar(features)

# Creating the 2 classes
class_0 = np.column_stack((np.random.multivariate_normal(mean_0, cov, n), [0 for _ in range(n)]))
class_1 = np.column_stack((np.random.multivariate_normal(mean_1, cov, n), [1 for _ in range(n)]))

# Random split into 70% train and 30% test
train_0, test_0 = random_split(class_0, 0.7)
train_1, test_1 = random_split(class_1, 0.7)

# Combining the train and test sets of the 2 classes
train, test = np.vstack((train_0,train_1)), np.vstack((test_0,test_1))

# print mean_0
# print mean_1
# print cov

# Save data
save_data_csv("DS1-train.csv",train)
save_data_csv("DS1-test.csv",test)
