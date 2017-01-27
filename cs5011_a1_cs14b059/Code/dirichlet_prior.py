import numpy as np
from math import log

# Training the MLE for multinomial likelihood with Dir(params) prior
# Finding the conditional probabilities of word given class
def train_multinomial_nb_dir(train_set, train_label, classes, params):
    nc = len(classes)
    v = len(train_set[0])

    prior = [0.0 for _ in range(nc)]
    class_data = [[] for _ in range(nc)]
    cond_prob = [[0.0 for _ in range(nc)] for _ in range(v)]

    for i in range(len(train_label)):
        class_data[train_label[i]].append(train_set[i])

    sigma_alpha = np.sum(params)

    for c in classes:
        denom = float(np.sum(class_data[c]) + sigma_alpha)
        numer = map(float, np.sum(class_data[c], axis=0))
        prior[c] = float(train_label.count(c)) / len(train_label)
        for t in range(v):
            cond_prob[t][c] = (numer[t] + params[t]) / denom

    return v,prior,cond_prob

# Applying the BPE model to a given data point
# Returning the class with maximum bayesian estimate for the
# given document
def apply_multinomial_nb_dir(classes, v, prior, cond_prob, doc):
    score = [0.0 for _ in range(len(classes))]
    w = np.nonzero(doc)[0]
    for c in classes:
        # score[c] = log(prior[c])
        for t in w:
            score[c] += log(cond_prob[t][c])
    return np.argmax(score), score

# Applying the BPE model to a given data set
# Returning the class with maximum bayesian estimate for the
# every given document
def predict_multinomial_nb_dir(train_set, train_label, test_set, params):
    classes = [0, 1]
    v, prior, cond_prob = train_multinomial_nb_dir(train_set, train_label, classes, params)
    y_pred = []
    y_score = []
    for d in test_set:
        a, b = apply_multinomial_nb_dir(classes, v, prior, cond_prob, d)
        y_pred.append(a)
        y_score.append(b)
    return np.array(y_pred), np.array(y_score)
