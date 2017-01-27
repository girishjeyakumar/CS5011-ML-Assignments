import pandas as pd
from utilities import data_path
import numpy as np

filename = "communities.csv"

# Reading data from csv
df = pd.read_csv(data_path + filename, header=None)

# Dropping the non-predective columns
df.drop([1, 2, 3, 4],axis=1,inplace=True)

# Looping through all columns and filling missing values
# with the mean of that column + some small random uniform noise
for i in df.axes[1]:
    if df[i].notnull().all(): continue
    m = df[i].mean()
    for j in range(len(df[i])):
        c = m+np.random.uniform(-1, 1)*(10**-2)
        c -= int(c)
        df[i][j] = c

# Saving data
df.to_csv(data_path+'communities_cleaned.csv',header=False,index=False)