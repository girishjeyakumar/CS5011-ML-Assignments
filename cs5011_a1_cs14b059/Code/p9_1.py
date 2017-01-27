from PIL import Image
import glob
from utilities import class_stats,save_data_csv
from sklearn import linear_model
import numpy as np

# Binning of pixel values to generate features
def features(data):
    rows = len(data)
    cols = len(data[0])
    all_features = []
    for j in xrange(cols):
        bins = [0]*32
        for i in xrange(rows):
            bins[data[i][j]/8] += 1
        all_features.extend(bins)
    return all_features


# Creating the train and test set from the images in DS3 dataset

splits = ["Test", "Train"]
classes = ["mountain", "forest"]
labels = {}
labels["mountain"] = -1
labels["forest"] = 1
X_train = []
y_train = []
X_test = []
y_test = []

for split in splits:
    for clas in classes:
        for filename in glob.glob(clas + "/" + split + '/*.jpg'):
            im=Image.open(filename)
            data = im.getdata()
            if split=='Train':
                X_train.append(features(data))
                y_train.append(labels[clas])
            else:
                X_test.append(features(data))
                y_test.append(labels[clas])

# Combining the classes with features to create the train and test dataset
train, test = np.column_stack((X_train,y_train)), np.column_stack((X_test,y_test))

# Saving data
save_data_csv("DS2-train.csv",train)
save_data_csv("DS2-test.csv",test)

# Fitting a Logistic Regression model
logreg = linear_model.LogisticRegression()
logreg.fit(X_train,y_train)

# Prediction
y_pred = logreg.predict(X_test)

# Printing out results
class_stats(y_pred,y_test,2)
