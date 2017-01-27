import glob
import csv
import os

data_path = "./Clustering_Data/"
csv_path = "./Data_CSV/"
arff_path = "./Data_ARFF/"


def save_as_arff(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    txt_file = file_path

    data = []
    classes = set()

    with open(txt_file, 'r') as file:
        for line in file:
            row = line.strip().split('\t')
            data.append(row)
            classes.add(int(row[2]))

    arff_file = open(arff_path + filename + '.arff', 'w')
    arff_file.write('@relation ' + filename + '\n\n')

    arff_file.write("@attribute x numeric\n")
    arff_file.write("@attribute y numeric\n")

    arff_file.write("@attribute class {" + ','.join(map(str,classes)) + "}")

    arff_file.write('\n\n@data\n\n')

    for d in data:
        arff_file.write(','.join(d) + "\n")

    arff_file.close()


def save_as_csv(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]

    txt_file = file_path
    csv_file = csv_path + filename + ".csv"

    for line in open(txt_file):
        print line.strip().split('\t')
    in_txt = csv.reader(open(txt_file, "rb"), delimiter='\t')
    out_csv = csv.writer(open(csv_file, 'wb'))

    out_csv.writerows(in_txt)


for file_path in glob.glob(data_path + '*.txt'):
    save_as_arff(file_path)
