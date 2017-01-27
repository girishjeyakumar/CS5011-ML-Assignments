import matplotlib.pyplot as plt

plots_path = "./plots/"

def plot_points(x,y,xlabel,ylabel,filename):
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plots_path+filename)
    plt.close()
    

def cluster_purity(clusterer,data,classes,debug=False):

    l = len(list(data))

    no_of_clusters = clusterer.number_of_clusters
    no_of_classes = len(set(classes))

    confusion_matrix = [[0 for _ in range(no_of_classes)] for _ in range(no_of_clusters)]

    for i in range(l):

        # Getting cluster label
        inst = data.get_instance(i)

        # Getting actual class label
        c = int(classes[i])

        try:
            cl = clusterer.cluster_instance(inst)
            confusion_matrix[cl][c] += 1
            
        except Exception as e:
            continue

    numer = 0

    for cl in confusion_matrix:
        numer += max(cl)

    purity = float(numer)/l

    if debug is True:
        print confusion_matrix

    return purity




