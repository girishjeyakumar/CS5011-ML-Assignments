from utilities import load_data_csv
from sklearn import linear_model
from sklearn.decomposition import PCA
from utilities import plot_3d,plot_2d_model,class_stats

def pca_projected_X(X,n):
    pca = PCA(n_components=n)
    return pca.fit_transform(X)

# Loading train data and projecting into derived space
X_train = load_data_csv("DS3/train.csv")
X_train_proj = pca_projected_X(X_train,1)
y_train = load_data_csv("DS3/train_labels.csv")

# Loading test data and projecting into derived space
X_test = load_data_csv("DS3/test.csv")
X_test_proj = pca_projected_X(X_test,1)
y_test = load_data_csv("DS3/test_labels.csv")

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
print list(y_test).count(0),list(y_test).count(1)
plot_3d(X_test,y_test)
class_stats(y_pred,y_test,2)
plot_2d_model(X_test_proj,y_test,lr.coef_[0],lr.intercept_)



