import weka.core.jvm as jvm
from weka.core.converters import Loader
from models import *

jvm.start(packages=True)

data_dir = "./Data_ARFF/"

# Loading the data
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + "jain.arff")

# Obtaining the class labels
classes = data.values(2)

# Deleting the class labels from the data
data.delete_last_attribute()

# Running DBScan
min_points_list = [16,17,18,19,20,21,22,23,24]
e_list = [0.07,0.08,0.09,0.1,0.11]
expr_DBScan(data,classes,"Jain",e_list,min_points_list)

jvm.stop()