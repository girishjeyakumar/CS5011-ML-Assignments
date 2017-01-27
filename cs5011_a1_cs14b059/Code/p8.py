from utilities import load_data_csv
from sklearn import linear_model
from sklearn.lda import LDA
from utilities import plot_3d,plot_2d_model,class_stats

def lda_projected_X(X,y,n):
    pca = LDA(n_components=n)
    return pca.fit_transform(X,y)

# Loading train data and projecting into derived space
X_train = load_data_csv("DS3/train.csv")
y_train = load_data_csv("DS3/train_labels.csv")
X_train_proj = lda_projected_X(X_train,y_train,1)

# Loading test data and projecting into derived space
X_test = load_data_csv("DS3/test.csv")
y_test = load_data_csv("DS3/test_labels.csv")
X_test_proj = lda_projected_X(X_test,y_test,1)

# Re-encoding output
y_train[y_train == 1] = 0
y_train[y_train == 2] = 1
y_test[y_test == 1] = 0
y_test[y_test == 2] = 1

# Fitting model
lr = linear_model.LinearRegression()
lr.fit(X_train_proj, y_train)

# Prediction
y_pred_actual = lr.predict(X_test_proj)
y_pred = y_pred_actual.copy()

# Thresholding
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1

# Plotting and printing out results
plot_3d(X_test,y_test)
class_stats(y_pred,y_test,2)
plot_2d_model(X_test_proj,y_test,lr.coef_[0],lr.intercept_)



