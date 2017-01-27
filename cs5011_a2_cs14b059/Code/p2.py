import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from utilities import class_stats, save_data_csv, load_data_csv,plot_points

nn_input_dim = 96  # input layer dimensionality
nn_output_dim = 4  # output layer dimensionality

pho = 0.01  # learning rate for gradient descent
# gamma = 0.1  # regularization strength


# gamma - 10, iters - 15000

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return np.multiply(sigmoid(x), (1.0 - sigmoid(x)))


def d_sigmoid(x):
    return np.multiply(x, (1 - x))


def get_data():
    # Load train and test set from csv
    train = load_data_csv("DS2-train.csv")
    test = load_data_csv("DS2-test.csv")

    # Getting the number of features
    features = len(train[0]) - 1

    # Splitting into features and classification
    X_train, y_train = np.hsplit(train, [features])
    X_test, y_test = np.hsplit(test, [features])

    X_train = scale(X_train)
    X_test = scale(X_test)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return X_train, y_train, X_test, y_test


def calculate_loss(model, X, y):
    n = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss
    error = 0.5 * np.sum(np.square(y - probs))

    # Add regulatization term to loss
    error += gamma * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return error


def predict(X, model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)


def train_model(X, y, nn_hdim, classes, num_passes=20000, print_loss=True):
    n = len(X)
    # np.random.seed(0)

    epsilon = (np.sqrt(6)) / np.sqrt((nn_input_dim + nn_hdim))
    W1 = (2 * epsilon * np.random.randn(nn_input_dim, nn_hdim) - epsilon)
    b1 = (2 * epsilon * np.random.randn(1, nn_hdim) - epsilon)

    epsilon = (np.sqrt(6)) / np.sqrt((nn_output_dim + nn_hdim))
    W2 = (2 * epsilon * np.random.randn(nn_hdim, nn_output_dim) - epsilon)
    b2 = (2 * epsilon * np.random.randn(1, nn_output_dim) - epsilon)

    # print W1
    model = {}

    vy = []
    for i in range(classes):
        vy.append((np.ones(n) * i == y).astype(int))

    y = np.column_stack(vy)

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Back Propagation

        sigma_pi_2 = np.sum(np.square(probs), axis=1, keepdims=True)
        sigma_pk_yk = np.sum(np.multiply(probs, y), axis=1, keepdims=True)
        delta3 = np.multiply(probs, (probs - y + sigma_pk_yk - sigma_pi_2))

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.multiply(delta3.dot(W2.T), d_sigmoid(a1))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms 
        dW2 += gamma * W2
        dW1 += gamma * W1

        # Gradient descent parameter update
        W1 += -pho * dW1
        b1 += -pho * db1
        W2 += -pho * dW2
        b2 += -pho * db2

        # Assigning new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))

    return model

acc_train = []
acc_test = []

def main():
    global nn_input_dim, nn_output_dim, gamma

    iters = 15000
    hl_nodes = 10
    gamma_trials = [0.01, 0.1, 0, 1, 10, 100]

    X_train, y_train, X_test, y_test = get_data()

    for gamma in gamma_trials:

        print "\n-------For gamma = %f and number of hidden layers = %d----------\n" %(gamma,hl_nodes)

        model = train_model(X_train, y_train, hl_nodes, 4, iters,False)

        t = predict(X_train, model)
        f1 = class_stats(t, y_train, 4)
        acc_train.append(f1)

        y_pred = predict(X_test, model)
        f1 = class_stats(y_pred, y_test, 4)
        acc_test.append(f1)

    plot_points(gamma_trials,acc_train,'gamma','accuray')
    plot_points(gamma_trials,acc_test,'gamma','accuracy')

if __name__ == "__main__":
    main()
