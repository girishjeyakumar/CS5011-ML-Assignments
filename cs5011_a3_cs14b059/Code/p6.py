import weka.core.jvm as jvm
from weka.core.converters import Loader
from utilities import plot_points
from models import *

jvm.start(packages=True)

data_dir = "./Data_ARFF/"

# Loading the data
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + "D31.arff")

# Obtaining the class labels
classes = data.values(2)

# Deleting the class labels from the data
data.delete_last_attribute()

print "---------- Performing K-Means for k ranging from 1 to 50 ------------"

purity_values = []

for k in range(1,50):
    purity = perform_KMeans(data,classes,k)
    purity_values.append(purity)

k_values = [i for i in range(1,51)]

plot_points(k_values,purity_values,"k","KMeans_Purity","p6_KMeans")


print "---------- Performing DBScan ------------"

min_points_list = [120,130,140,150,160,170,180,190,200]
e_list = [0.08,0.1,0.2]

expr_DBScan(data,classes,"D31",e_list,min_points_list)


print "---------- Performing Hierarchical Clustering for k ranging from 28 to 36 ------------"

purity_values = []

s = "WARD "

for k in range(28,36):

    purity = perform_HC(data,classes,k,'WARD')
    purity_values.append(purity)

k_values = [i for i in range(28,36)]

plot_points(k_values,purity_values,"k","HC Purity","p6_HC")


jvm.stop()