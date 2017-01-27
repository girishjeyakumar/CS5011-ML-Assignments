from sklearn import linear_model

from utilities import *

# Alpha - regularization parameter
# Alpha = Lambda

# Alpha validation by training model on train data and checking performance
# on test data
def alpha_validation_1(alpha):
    total_rse = 0
    iterations=5
    print "Lambda: %f\n"%(alpha)
    for i in range(1, iterations+1):
        train = load_data_csv("CandC-train" + str(i) + ".csv")
        test = load_data_csv("CandC-test" + str(i) + ".csv")

        features = len(train[0]) - 1

        X_train, y_train = np.hsplit(train, [features])
        X_test, y_test = np.hsplit(test, [features])

        rlr = linear_model.Ridge(alpha=alpha,fit_intercept=True, normalize=False)
        rlr.fit(X_train, y_train)

        rse = np.mean((rlr.predict(X_test) - y_test) ** 2)

        # Printing out results of current model
        print"Split %d:\n"%(i)
        print "RSE: %f\n"%(rse)
        print "Coefficients: ", rlr.coef_[0]
        print "\nIntercept: %f\n" % (rlr.intercept_[0])
        total_rse = total_rse + rse

    mean_rse = total_rse / iterations

    print "Mean RSE: %f\n" %(mean_rse)
    print "-------------------------------------"
    return mean_rse

# Alpha validation by training model on 75% of train data and checking performance
# on the other 25% of train data
def alpha_validation_2(alpha):
    total_rse = 0
    iterations = 5

    for i in range(1, iterations+1):
        train = load_data_csv("CandC-train" + str(i) + ".csv")

        l = len(train)
        trn,val = train[0:3*(l/4)],train[3*(l/4):l]

        features = len(train[0]) - 1

        # Splitting the features and column to be predicted
        X_train, y_train = np.hsplit(trn, [features])
        X_val, y_val = np.hsplit(val, [features])

        rlr = linear_model.Ridge(alpha=alpha, fit_intercept=True, normalize=False)
        rlr.fit(X_train, y_train)

        # Residual Error
        rse = np.mean((rlr.predict(X_val) - y_val) ** 2)

        total_rse = total_rse + rse

    mean_rse = total_rse / iterations

    print "Mean validation set RSE for alpha= %f: " % (alpha), mean_rse

    return mean_rse

# Trying different values of lambda and plotting the
# RSE vs Lambda graph and choosing the ideal lambda
start = 0.5
step = 0.5

min_mean_rse = alpha_validation_1(start)
min_lambd = start

# Collecting data for plot
x = []
y = []

x.append(start)
y.append(min_mean_rse)

for i in range(1, 20):
    lambd = start + i * step
    mean_rse = alpha_validation_1(lambd)

    # Collecting data for plot
    x.append(lambd)
    y.append(mean_rse)

    if mean_rse < min_mean_rse:
        min_mean_rse = mean_rse
        min_lambd= lambd

plot_points(x,y,"Lambda","RSE")

# Printing out the best fit lambda and results of the corresponding model
print "Best fit lambda: ", min_lambd

alpha_validation_1(1.5)



