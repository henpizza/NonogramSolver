#!/usr/bin/env python3
"""
Created on Sun Jul  7 17:47:40 2024
"""

import numpy as np
from itertools import product,groupby
from os import chdir,listdir
from os.path import isfile
import pandas as pd
from pprint import pp
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sys import exit
import matplotlib.pyplot as plt

chdir("nonogram")

from auxiliary_module import NonogramFrame,convert_to_nonogram_frame


#files = [f for f in listdir('.') if isfile(f)] 

data = NonogramFrame()
#target = pd.DataFrame(columns = ['target'])



for filename in range(1,100):
    data = pd.concat([data,convert_to_nonogram_frame(filename)])
    
data.to_csv('training_data.csv',index=False)
data = pd.read_csv('training_data.csv')

#print(data[['len_col_std']].describe())
scaler_0_1 = MinMaxScaler()
scaler = MinMaxScaler(feature_range=(-1,1))
scaler = StandardScaler()
#data = pd.DataFrame(scaler_0_1.fit_transform(data),columns=data.columns,index=data.index)
attribs_0_1 = ['shape_rows','shape_cols',
                               'total_row', 'total_col', 'col', 'row',
                               #'min_len_col', 'max_len_col', #'len_col_diff',
                               #'min_len_row', 'max_len_row', #'len_row_diff',
                               'num_row', 'num_col',
                               #'len_row_avg','len_row_std',
                               #'len_col_avg','len_col_std',
                               #'left','right','up','down',
                               ]
#print(data[['len_col_std']].describe())
data.hist()
plt.show()
for k in attribs_0_1:
    print(data[[k]].describe())
    print()
data[attribs_0_1] = scaler.fit_transform(data[attribs_0_1])
for k in attribs_0_1:
    print(data[[k]].describe())
    print()
data.hist()
plt.show()


data.to_csv('training_data.csv',index=False)
    
            
    
