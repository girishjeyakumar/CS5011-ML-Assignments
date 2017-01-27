import glob
from utilities import class_stats, plot_pr_curve
from multinomial_nb import predict_multinomial_nb
from bernoulli_nb import predict_bernoulli_nb
from dirichlet_prior import predict_multinomial_nb_dir
from beta_prior import predict_bernoulli_nb_beta
import numpy as np

data_path = "./data/Q10/"

# Generating word vector for a given file
def word_vector(filename):
    lines = [line.strip() for line in open(filename)]
    w_vector = []
    wi_vector = []
    for line in lines:
        words = line.split(' ')
        w_vector += words
    for i in range(len(w_vector)):
        word = w_vector[i]
        try:
            index = int(word)
        except ValueError:
            continue
        wi_vector.append(index)
    return wi_vector

# Generating frequency vector for a given file
def frequency_vector(filename, k):
    f_vector = [0 for _ in range(k + 1)]
    w_vector = word_vector(filename)
    for word in w_vector:
        f_vector[word] += 1
    return f_vector[1::]


# Finding size of vocabulary
def find_k(train):
    k = 0
    for part in train:
        for filename in glob.glob(data_path + "part" + str(part) + '/*.txt'):
            w_vector = word_vector(filename)
            k = max(k, max(w_vector))
    return k


# Creating the p x k trainMatrix
def train_matrix(train, k):
    train_set = []
    train_label = []
    for part in train:
        for filename in glob.glob(data_path + "part" + str(part) + '/*legit*.txt'):
            f_vector = frequency_vector(filename, k)
            train_set.append(f_vector)
            train_label.append(1)
        for filename in glob.glob(data_path + "part" + str(part) + '/*spmsg*.txt'):
            f_vector = frequency_vector(filename, k)
            train_set.append(f_vector)
            train_label.append(0)
    return train_set, train_label

# Creating the r x k testMatrix
def test_matrix(test, k):
    test_set = []
    test_label = []
    for part in test:
        for filename in glob.glob(data_path + "part" + str(part) + '/*legit*.txt'):
            f_vector = frequency_vector(filename, k)
            test_set.append(f_vector)
            test_label.append(1)
        for filename in glob.glob(data_path + "part" + str(part) + '/*spmsg*.txt'):
            f_vector = frequency_vector(filename, k)
            test_set.append(f_vector)
            test_label.append(0)
    return test_set, test_label

# Creating the trainMatrix,trainLabel,testMatrix and testLabel
def create_datasets(start):

    test = [start, start+1]
    train = []
    for i in range(1,11):
        if i in test:continue
        train.append(i)
    k = find_k(train)
    train_set, train_label = train_matrix(train, k)
    test_set, test_label = test_matrix(test, k)
    return train_set, train_label, test_set, test_label


def perform_multinomial_nb():
    train_set, train_label, test_set, test_label = create_datasets(1)
    y_pred, y_score = predict_multinomial_nb(train_set, train_label, test_set)
    plot_pr_curve(test_label, y_score)
    class_stats(y_pred, test_label, 2)

def perform_bernoulli_nb():
    train_set, train_label, test_set, test_label = create_datasets(1)
    y_pred, y_score = predict_bernoulli_nb(train_set, train_label, test_set)
    plot_pr_curve(test_label, y_score)
    class_stats(y_pred, test_label, 2)


def perform_bernoulli_nb_beta(params):
    acc_val = []
    for s in [1, 3, 5, 7, 9]:
        train_set, train_label, test_set, test_label = create_datasets(s)
        y_pred, y_score = predict_bernoulli_nb_beta(train_set, train_label, test_set, params)
        plot_pr_curve(test_label, y_score)
        acc, _ = class_stats(y_pred, test_label, 2)
        acc_val.append(acc)
    print sum(acc_val) / 5

def perform_multinomial_nb_dir(params):
    acc_val = []
    for s in [1, 3, 5, 7, 9]:
        train_set, train_label, test_set, test_label = create_datasets(s)
        y_pred, y_score = predict_multinomial_nb_dir(train_set, train_label, test_set, params)
        plot_pr_curve(test_label, y_score)
        acc, _= class_stats(y_pred, test_label, 2)
        acc_val.append(acc)
    print sum(acc_val)/5


'''
For printing out various results
'''
perform_multinomial_nb()
perform_bernoulli_nb()

a_max = 0
param_max = []

params = [[0.1,0.6],[1,0.1],[4,4],[10,5]]
for x in params:
    a = perform_bernoulli_nb_beta(x)
    if a>a_max:
        a_max = a
        param_max =x
    print a_max,param_max

perform_bernoulli_nb_beta(param_max)

a_max = 0
param_max = []

for i in range(4):
    params = np.random.rand(24747)
    a = perform_multinomial_nb_dir(params)
    if a>a_max:
        a_max = a
        param_max =params

perform_multinomial_nb_dir(param_max)
