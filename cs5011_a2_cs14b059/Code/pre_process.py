from PIL import Image
import glob
from utilities import class_stats,save_data_csv,load_data_csv
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


# Creating the train and test set from the images in DS2 dataset

splits = ["Test", "Train"]
classes = ["coast", "forest","insidecity","mountain"]

labels = {}
labels["coast"] = 0
labels["forest"] = 1
labels["insidecity"] = 2
labels["mountain"] = 3

X_train = []
y_train = []
X_test = []
y_test = []

for split in splits:
    for clas in classes:
        for filename in glob.glob("data/" + clas + "/" + split + '/*.jpg'):
            print filename
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
