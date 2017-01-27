from sklearn import linear_model

from utilities import *

data = load_data_csv("communities_cleaned.csv")

iterations = 5

# Creating and saving random 80-20 splits
for i in range(1,iterations+1):
    train,test = random_split(data,0.8)
    save_data_csv("CandC-train"+ str(i) +".csv",train)
    save_data_csv("CandC-test" + str(i) +".csv", test)

total_rse = 0

# Finding the residual error for each value, on test data, averaged over 5 different 80-20
# splits and printing out the results
for i in range(1,iterations+1):
    train = load_data_csv("CandC-train"+ str(i) +".csv")
    test = load_data_csv("CandC-test" + str(i) +".csv")

    features = len(train[0]) - 1

    # Splitting the features and column to be predicted
    X_train, y_train = np.hsplit(train, [features])
    X_test, y_test = np.hsplit(test, [features])

    lr = linear_model.LinearRegression(fit_intercept=True,normalize=False)
    lr.fit(X_train, y_train)

    rse = np.mean((lr.predict(X_test) - y_test) ** 2)
    print "For split %d:\n"%(i)
    print "RSE: %f\n"%(rse)
    print lr.coef_[0]
    print "\nIntercept: %f\n"%(lr.intercept_[0])
    total_rse = total_rse+rse

print "Mean RSE: ",total_rse/iterations


