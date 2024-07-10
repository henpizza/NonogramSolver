#!/usr/bin/env python3
"""
Created on Mon Jul  8 13:29:51 2024
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

class NonogramFrame(pd.DataFrame):
    def __init__(self):
        super().__init__(columns = ('shape_rows','shape_cols',
                                       'total_row', 'total_col', 'col', 'row',
                                       #'min_len_col', 'max_len_col', #'len_col_diff',
                                       #'min_len_row', 'max_len_row', #'len_row_diff',
                                       'num_row', 'num_col',
                                       #'len_row_avg','len_row_std',
                                       #'len_col_avg','len_col_std',
                                       #'left','right','up','down',
                                       'target'))


def convert_to_nonogram_frame(filename: str):
    cur_dir = getcwd()
    chdir("data")
    
    data = NonogramFrame()
    
    shape_rows = []
    shape_cols = []
    total_row = []
    total_col = []
    col = []
    row = []
    min_len_col = []
    max_len_col = []
    len_col_diff = []
    min_len_row = []
    max_len_row = []
    len_row_diff = []
    num_row = []
    num_col = []
    len_row_avg = []
    len_row_std = []
    len_col_avg = []
    len_col_std = []
    left = []
    right = []
    up = []
    down = []
    target_list = []
    
    rows = []
    cols = []
    shape = [0,0]
    with open(f'{filename}.dat','r') as file:
        shape = tuple(map(int,file.readline().split(',')))
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
    for i,j in product(range(len(rows)),range(len(cols))):
        shape_rows.append(shape[0])
        shape_cols.append(shape[1])
        total_row.append(sum(rows[i]))
        total_col.append(sum(cols[j]))
        row.append(i)
        col.append(j)
        min_len_row.append(min(rows[i]))
        min_len_col.append(min(cols[j]))
        max_len_row.append(max(rows[i]))
        max_len_col.append(max(cols[j]))
        len_row_diff.append(max(rows[i])-min(rows[i]))
        len_col_diff.append(max(cols[j])-min(cols[j]))
        num_row.append(len(rows[i]))
        num_col.append(len(cols[j]))
        len_row_avg.append(np.average(rows[i]))
        len_row_std.append(np.std(rows[i]))
        len_col_avg.append(np.average(cols[j]))
        len_col_std.append(np.std(cols[j]))
        left.append(map_entry_to_value(nonogram_target, (i,j-1)))
        right.append(map_entry_to_value(nonogram_target, (i,j+1)))
        up.append(map_entry_to_value(nonogram_target, (i-1,j)))
        down.append(map_entry_to_value(nonogram_target, (i+1,j)))
        target_list.append(nonogram_target[i,j])
        
    data['shape_rows'] = shape_rows
    data['shape_cols'] = shape_cols
    data['total_row'] = total_row
    data['total_col'] = total_col
    data['row'] = row
    data['col'] = col
    #data['min_len_row'] = min_len_row
    #data['min_len_col'] = min_len_col
    #data['max_len_row'] = max_len_row
    #data['max_len_col'] = max_len_col
    #data['len_row_diff'] = len_row_diff
    #data['len_col_diff'] = len_col_diff
    data['num_row'] = num_row
    data['num_col'] = num_col
    #data['len_row_avg'] = len_row_avg
    #data['len_row_std'] = len_row_std
    #data['len_col_avg'] = len_col_avg
    #data['len_col_std'] = len_col_std
    #data['left'] = left
    #data['right'] = right
    #data['up'] = up
    #data['down'] = down
    #target['target'] = target_list
    data['target'] = target_list
    
    chdir(cur_dir)
    
    return data

def map_entry_to_value(nonogram,entry_coord):
    value = 0
    with suppress(IndexError):
        row = entry_coord[0]
        col = entry_coord[1]
        value = nonogram[row][col]
        if value == 0: value = -1
    return value
