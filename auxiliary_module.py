#!/usr/bin/env python3
"""
Created on Wed Jul 10 11:05:13 2024
"""

import pandas as pd
from os import chdir,getcwd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from contextlib import suppress
from itertools import product,groupby
from random import choice




FILLED = 1
NOT_FILLED = 0
ROW_ID = 1
COL_ID = 0


class NonogramFrame(pd.DataFrame):
    def __init__(self, shape: tuple):
        super().__init__(columns = (
                                       ))
        for k in range(1,shape[0]+1):
            self['row_' + str(k) + '_total'] = np.nan
            self['row_' + str(k) + '_spaces'] = np.nan
        for k in range(1,shape[1]+1):
            self['col_' + str(k) + '_total'] = np.nan
            self['col_' + str(k) + '_spaces'] = np.nan
            
        self['target'] = np.nan


def convert_generated_data_to_nonogram_frame(filename_list: list, shape: tuple, entry_coord: tuple):
    '''Jestli tohle bude fungovat, musis to predelat.'''
    
    cur_dir = getcwd()
    chdir("data")
    
    data = NonogramFrame(shape)
    

    row_total = [[] for _ in range(shape[0])]
    row_spaces = [[] for _ in range(shape[0])]
    col_total = [[] for _ in range(shape[1])]
    col_spaces = [[] for _ in range(shape[1])]
    target_list = []
    
    for filename in filename_list:
        rows = []
        cols = []
        
        with open(f'{filename}.dat','r') as file:
            file.readline()
            file.readline()
            while('-' not in (inp := file.readline())):
                if (len(inp) > 1):
                    rows.append(list(map(int,inp.split(','))))
                else:
                    rows.append([0])
            while('-' not in (inp := file.readline())):
                if (len(inp) > 1):
                    cols.append(list(map(int,inp.split(','))))
                else:
                    cols.append([0])
                        
        nonogram_target = np.loadtxt(f'{filename}.target',dtype=int)
        
        row = entry_coord[0]-1
        col = entry_coord[1]-1
        
        for k in range(0,shape[0]):
            row_total[k].append(sum(rows[k]))
            row_spaces[k].append(len(rows[k]))
        for k in range(0,shape[1]):
            col_total[k].append(sum(cols[k]))
            col_spaces[k].append(len(cols[k]))
            
        target_list.append(nonogram_target[row,col])

    for k in range(0,shape[0]):
        data['row_' + str(k+1) + '_total'] = row_total[k]
        data['row_' + str(k+1) + '_spaces'] = row_spaces[k]
    for k in range(0,shape[1]):
        data['col_' + str(k+1) + '_total'] = col_total[k] 
        data['col_' + str(k+1) + '_spaces'] = col_spaces[k]
    data['target'] = target_list

    chdir(cur_dir)
    
    return data







def nonogram_data_generate(shape: tuple, num: int = 500):
    cur_dir = getcwd()
    chdir("data")
    for n in range(num):
        
        
        rows = shape[0]
        columns = shape[1]
    
        nonogram = np.full((rows,columns),0)
    
        for i,j in product(range(rows),range(columns)):
            nonogram[i,j] = choice([FILLED,NOT_FILLED])
            
        rows_out = []
        for row in nonogram:
            groups = groupby(row)
            result = [sum(1 for _ in group) for label, group in groups if label == FILLED]
            rows_out.append(result)
            
        columns_out = []
        for column in nonogram.T:
            groups = groupby(row)
            result = [sum(1 for _ in group) for label, group in groups if label == FILLED]
            columns_out.append(result)
        
        with open(f"{n+1}.dat","w") as out:
            print(f'{rows},{columns}',file=out)
            print('-',file=out)
            for row in rows_out:
                print(*row,sep=',',file=out)
            print('-',file=out)
            for col in columns_out:
                print(*col,sep=',',file=out)
            print('-',file=out)
        
        with open(f'{n+1}.target',"w") as out:
            np.savetxt(out,nonogram,fmt="%i")
    chdir(cur_dir)
    
