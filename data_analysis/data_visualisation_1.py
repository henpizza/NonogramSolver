'''
This visualisation code was made for the commit no. 9.
Produces a scatter plot for 2 features.

Run with Jupyter.
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from collections import Counter


# Adding parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram


shape = (10,10)
size = shape[0] * shape[1]
n_dimensions = (shape[0]+shape[1])*2
num = 100

nonogram = make_empty_nonogram(shape)



generate_nonogram_data(shape,num=num,template=nonogram)

data,targets = convert_generated_data_to_data_frames(range(1,num), shape)

data = data.to_numpy()

col_1 = data[:,0]
col_2 = data[:,2]

cols = np.vstack((col_1,col_2))
cols = list(map(tuple,cols.T))
cols = Counter(cols)
print(cols)
for xy in cols:
    plt.scatter(xy[0],xy[1],cols[xy]*10,c='peru')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(max(1,shape[0]//10)))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(max(1,shape[1]//10)))
plt.show()


