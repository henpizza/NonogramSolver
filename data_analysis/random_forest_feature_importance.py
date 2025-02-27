'''
This test was made for the commit no. 8.
It shows, with the help of a random forest, the importance of the features
row/col_spaces and row/col_total.
Results: All rows and columns that are not the field's row or column
are almost equally important, but their influence is not negligible.

Run with Jupyter.
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import set_config
set_config(transform_output = "pandas")

from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler



# Adding parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram


shape = (5,5)
size = shape[0] * shape[1]
num = 300

scaler = StandardScaler()



nonogram = make_empty_nonogram(shape)

model = RandomForestClassifier(n_estimators=num//2, max_leaf_nodes=30, n_jobs=-1)

importance_list = [[] for _ in range(max(shape[0],shape[1]))]

generate_nonogram_data(shape,num=num,template=nonogram)

data,targets = convert_generated_data_to_data_frames(range(1,num), shape)
data = scaler.fit_transform(data)






for i,j in product(range(shape[0]),range(shape[1])):
    target = targets[f'target_{i+1}_{j+1}'].copy()
    
    try: # all targets might have the same label (although it is not probable)
        model.fit(data,target)
    except ValueError:
        continue

    # Printing 10 most important features for a field
    print(i+1,j+1)
    l = []
    for score, name in zip(model.feature_importances_, data.columns):
        l.append((score,name))
    l.sort(key=lambda x: x[0], reverse=True)
    l = l[:10]
    for score,name in l:
        print(round(score,2),name)
    print()

    # Average score plot
    for score, name in zip(model.feature_importances_, data.columns):
        name = name.split('_')
        row_col_num = int(name[1])
        if 'row' in name[0]:
            importance_list[abs(row_col_num-i-1)].append(score)
        elif 'col' in name[0]:
            importance_list[abs(row_col_num-j-1)].append(score)
        else:
            print('ERROR')
            exit(1)
    
importance_list = list(map(np.average,importance_list))
x = range(len(importance_list))
plt.plot(x,importance_list,c='peru')
plt.xlabel('Distance of row/column from a field')
plt.ylabel('Average score')
plt.suptitle('Feature importance',y=0.96,fontsize=12)
plt.title('(according to a random forest classifier)',fontsize=8)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(max(1,len(x)//10)))
plt.show()