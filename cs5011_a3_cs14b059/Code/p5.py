import weka.core.jvm as jvm
from weka.core.converters import Loader
from models import *

jvm.start(packages=True)

# Running HC over a grid of k X link. Basically GridSearch for HC.
def expr_HC(data,classes,name):

	print "\n-------- Performing Heirarchical Clustering on %s ----------\n" %(name)

	actual_k = len(set(classes))
	print "Actual number of clusters = %d\n" %(actual_k)

	r = 3
	links = ['SINGLE','COMPLETE','AVERAGE','MEAN','CENTROID','WARD','ADJCOMPLETE','NEIGHBOR_JOINING']

	max_purity = 0.0
	best = (0.0,0.0,0.0)
	values = []

	for k in range(max(1,actual_k-r),actual_k+r+1):

		print "-------- k = %d --------" %(k)

		for link in links:

			purity = perform_HC(data,classes,k,link)
			print "Purity for link %s: %f" %(link,purity)

			values.append([link,k,purity])

			if purity>max_purity:
				best = (link,k,purity)
				max_purity = purity

	print best

data_dir = "./Data_ARFF/"

# Loading the data
loader = Loader(classname="weka.core.converters.ArffLoader")
path_based = loader.load_file(data_dir + "pathbased.arff")
spiral = loader.load_file(data_dir + "spiral.arff")
flame = loader.load_file(data_dir + "flame.arff")

# Obtaining the class labels
path_classes = path_based.values(2)
spiral_classes = spiral.values(2)
flame_classes = flame.values(2)

# Deleting the class labels from the data
path_based.delete_last_attribute()
spiral.delete_last_attribute()
flame.delete_last_attribute()

# Performing Hierarchical Clustering
expr_HC(path_based,path_classes,"Path-based")


# Performing DBScan

min_points_list = [1,2,4,8,16,32]
e_list = [0.1,0.2,0.4,0.8,1.6,3.2]

expr_DBScan(path_based,path_classes,"Path-based",e_list,min_points_list)

expr_DBScan(spiral,spiral_classes,"Spiral",e_list,min_points_list)

expr_DBScan(flame,flame_classes,"Flame",e_list,min_points_list)

jvm.stop()