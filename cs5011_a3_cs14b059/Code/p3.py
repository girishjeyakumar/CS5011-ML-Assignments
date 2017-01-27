import weka.core.jvm as jvm
from weka.core.converters import Loader
from utilities import plot_points
from models import *

jvm.start()

data_dir = "./Data_ARFF/"

# Loading the data
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + "R15.arff")

# Obtaining the class labels
classes = data.values(2)

# Deleting the class labels from the data
data.delete_last_attribute()

print "---------- Performing K-Means for k=8 ------------"

purity = perform_KMeans(data,classes,8)
print "Purity for k = 8 is %f" %(purity)


print "---------- Performing K-Means for k ranging from 1 to 20 ------------"

purity_values = []

for k in range(1,21):
    purity = perform_KMeans(data,classes,k)
    purity_values.append(purity)

k_values = [i for i in range(1,21)]

plot_points(k_values,purity_values,"K","Cluster Purity","p3_KMeans")

jvm.stop()