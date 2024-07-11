#!/usr/bin/env python3


import numpy as np
import pandas as pd

from itertools import groupby,product
# Spyder complains that it cannot reload the module if I use chdir
from os import chdir,getcwd,makedirs
from os.path import exists
from random import choice

# Pandas complains with PerformanceWarnings, but according to https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o,
# it is probably wrong.
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)




FILLED = 1
NOT_FILLED = 0
ROW_ID = 1
COL_ID = 0


class NonogramFrame(pd.DataFrame):
    """
    The class creates a pandas.DataFrame with the following columns:
        - row/col_i_total - The total number of filled fields in row/column i.
        - row/col_i_spaces - The total number of spaces

    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).

    """
    def __init__(self, shape: tuple):
        super().__init__(columns = ())
        for k in range(1,shape[0]+1):
            self['row_' + str(k) + '_total'] = np.nan
            self['row_' + str(k) + '_spaces'] = np.nan
        for k in range(1,shape[1]+1):
            self['col_' + str(k) + '_total'] = np.nan
            self['col_' + str(k) + '_spaces'] = np.nan


class TargetFrame(pd.DataFrame):
    """
    The class creates a pandas.DataFrame with the following columns:
        - target_i_j - The target in the row i and column j.

    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).

    """
    def __init__(self, shape: tuple):
        super().__init__(columns = ())
        for i,j in product(range(1,shape[0]+1),range(1,shape[1]+1)):
            self['target_' + str(i) + '_' + str(j)] = np.nan
            #print(self)
            #self = self.copy()
            #pd.concat([self, np.nan], names=[list(self),'target_' + str(i) + '_' + str(j)],axis='columns')
            #self.astype(int)



def convert_to_nonogram_frame(filename_list: list, shape: tuple) -> NonogramFrame:
    """
    Parameters
    ----------
    filename_list : list
        List of files with the str datatype.
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    NonogramFrame

    """
    cur_dir = getcwd()
    chdir("data")

    data = NonogramFrame(shape)


    row_total = [[] for _ in range(shape[0])]
    row_spaces = [[] for _ in range(shape[0])]
    col_total = [[] for _ in range(shape[1])]
    col_spaces = [[] for _ in range(shape[1])]

    for filename in filename_list:
        rows = []
        cols = []

        with open(filename,'r') as file:
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


        for k in range(0,shape[0]):
            row_total[k].append(sum(rows[k]))
            row_spaces[k].append(len(rows[k])+1)
        for k in range(0,shape[1]):
            col_total[k].append(sum(cols[k]))
            col_spaces[k].append(len(cols[k])+1)

    for k in range(0,shape[0]):
        data['row_' + str(k+1) + '_total'] = row_total[k]
        data['row_' + str(k+1) + '_spaces'] = row_spaces[k]
    for k in range(0,shape[1]):
        data['col_' + str(k+1) + '_total'] = col_total[k]
        data['col_' + str(k+1) + '_spaces'] = col_spaces[k]

    chdir(cur_dir)

    return data


def convert_to_target_frame(filename_list: list, shape: tuple) -> TargetFrame:
    """
    Parameters
    ----------
    filename_list : list
        List of files with the str datatype.
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    TargetFrame

    """
    cur_dir = getcwd()
    chdir("data")

    target = TargetFrame(shape)

    for filename in filename_list:
        nonogram_target = np.loadtxt(filename,dtype=int)
        nonogram_target = pd.DataFrame([nonogram_target.flatten()],
                                       columns=list(target))
        target = pd.concat([target,nonogram_target],copy=False,ignore_index=True)


    target = target.astype(int) # pd.concat converts int to float for some reason

    chdir(cur_dir)

    return target



def nonogram_data_generate(shape: tuple, num: int = 500) -> None:
    """
    Generates num nonograms with shape[0] rows and shape[1] columns.

    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
    num : int, optional
        The number of nonograms to generate. The default is 500.

    Returns
    -------
    None.

    """
    if not exists('data'):
        makedirs('data')

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

        with open(f"{n+1}.non","w") as out:
            for row in rows_out:
                print(*row,sep=',',file=out)
            print('-',file=out)
            for col in columns_out:
                print(*col,sep=',',file=out)
            print('-',file=out)

        with open(f'{n+1}.target',"w") as out:
            np.savetxt(out,nonogram,fmt="%i")
    chdir(cur_dir)


def convert_generated_data_to_data_frames(filename_list: list, shape: tuple) -> tuple[NonogramFrame,TargetFrame]:
    """
    A convenience function transforming generated nonograms (see also nonogram_data_generate)
    into pandas.DataFrame tables (or more precisely, its subclasses NonogramFrame and TargetFrame).

    Parameters
    ----------
    filename_list : list
        List of indices indicating which files generated by nonogram_data_generate to process.
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    tuple[NonogramFrame,TargetFrame]

    """
    nonogram_frame_filename_list = ['a'] * len(filename_list)
    target_frame_filename_list = ['a'] * len(filename_list)
    for k in range(len(filename_list)):
        nonogram_frame_filename_list[k] = f'{filename_list[k]}.non'
        target_frame_filename_list[k] = f'{filename_list[k]}.target'

    nonogram_frame = convert_to_nonogram_frame(nonogram_frame_filename_list, shape)
    target_frame = convert_to_target_frame(target_frame_filename_list, shape)

    return (nonogram_frame,target_frame)