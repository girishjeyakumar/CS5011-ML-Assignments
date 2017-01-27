from utilities import cluster_purity
from weka.clusterers import Clusterer

def perform_KMeans(data,classes,k):

    clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(k)])
    clusterer.build_clusterer(data)
    purity = cluster_purity(clusterer, data, classes)
    return purity

def perform_DBScan(data,classes,e,min_points):

    clusterer = Clusterer(classname="weka.clusterers.DBSCAN", options=["-E", str(e), "-M", str(min_points)])
    clusterer.build_clusterer(data)
    purity = cluster_purity(clusterer, data, classes)
    return purity

def perform_HC(data,classes,k,link):

    clusterer = Clusterer(classname="weka.clusterers.HierarchicalClusterer", options=["-N", str(k), "-L", link])
    clusterer.build_clusterer(data)
    purity = cluster_purity(clusterer, data, classes)
    return purity

# Running DBScan over a grid of e X minpoints. Basically GridSearch for DBScan.
def expr_DBScan(data,classes,name,e_list,min_points_list):

    print "\n-------- Performing DBScan on %s ----------\n" %(name)

    actual_k = len(set(classes))
    print "Actual number of clusters = %d\n" %(actual_k)

    max_purity = 0.0
    best = (0.0,0.0,0.0)

    values = []

    for e in e_list:
        for min_points in min_points_list:

            purity = perform_DBScan(data,classes,e,min_points)
            values.append([e,min_points,purity])

            if purity>max_purity:
                best = (e,min_points,purity)
                max_purity = purity

    return best