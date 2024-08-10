import numpy as np
import pandas as pd
import os

from itertools import groupby,product
from inspect import stack
from numba import njit
# Spyder complains that it cannot reload the module if I use chdir
from os import chdir,getcwd,makedirs
from os.path import exists
from random import choice
from sys import exit

# Pandas complains with PerformanceWarnings when using large DataFrames, but according to
# https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o,
# it is probably wrong.
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)




FILLED = 1
NOT_FILLED = 0
UNKNOWN = -1
ROW_ID = 1
COL_ID = 0
NONOGRAM_FILENAME_EXTENSION = '.non'
TARGET_FILENAME_EXTENSION = '.target'
# This variable is needed only by some functions,
# so it might be unreasonable to demand Windows or Linux.
if os.name == 'nt': # Windows
    DATA_DIRECTORY = 'data\\'
elif os.name == 'posix':
    DATA_DIRECTORY = 'data/'
else:
    print('ERROR: Unknown filesystem type detected. Are you using Linux or Windows?')
    exit(1)

# These global variable are set in the set_shape() function.
shape = None
size = None
n_dimensions = None # only for NonogramFrame2
# Maximum number of sequences in a row/column # only for NonogramFrame2
max_num_of_sequences_in_row = 0
max_num_of_sequences_in_column = 0



class NonogramFrame(pd.DataFrame):
    """
    (This class might be deprecated in the future.)

    The class creates a pandas.DataFrame with the following columns:
        - row/col_i_total - The total number of filled fields in row/column i.
        - row/col_i_spaces - The total number of spaces

    Parameters
    ----------
    shape : tuple, optional
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    """
    def __init__(self, shape: tuple = None):
        if (shape is None):
            shape = get_shape()
        super().__init__(columns = ())
        for k in range(1,shape[0]+1):
            self['row_' + str(k) + '_total'] = np.nan
            self['row_' + str(k) + '_spaces'] = np.nan
            #self['row_' + str(k) + '_max'] = np.nan # this does not seem to help
        for k in range(1,shape[1]+1):
            self['col_' + str(k) + '_total'] = np.nan
            self['col_' + str(k) + '_spaces'] = np.nan
            #self['col_' + str(k) + '_max'] = np.nan


class NonogramFrame2(pd.DataFrame):
    """
    The class creates a pandas.DataFrame with the following columns:
        - row/col_i_k - The length of the k-th sequence of 1s in the row or column i.

    Parameters
    ----------
    data : np.array, optional
        Data that will populate the DataFrame.
        An np.array is expected, but it can be any type that pandas.DataFrame accepts in its constructor.
        If not specified, an empty NonogramFrame2 will be returned.
    shape : tuple, optional
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    """
    def __init__(self, data: np.array = None, shape: tuple = None):
        if (shape is None):
            shape = get_shape()
        column_names_1 = ['row_' + str(k) + '_' + str(m)
            for k in range(1,shape[0]+1)
            for m in range(1,max_num_of_sequences_in_row+1)]
        column_names_2 = ['col_' + str(k) + '_' + str(m)
            for k in range(1,shape[1]+1)
            for m in range(1,max_num_of_sequences_in_column+1)]
        if data is None:
            super().__init__(columns = ())
            for name in column_names_1:
                self[name] = np.nan
            for name in column_names_2:
                self[name] = np.nan
        else:
            column_names = column_names_1 + column_names_2
            super().__init__(data,columns=column_names)


class TargetFrame(pd.DataFrame):
    """
    The class creates a pandas.DataFrame with the following columns:
        - target_i_j - The target in the row i and column j.

    The fields are ordered in a row-by-row manner in the resulting TargetFrame.

    Parameters
    ----------
    data : np.array, optional
        Data that will populate the DataFrame.
        An np.array is expected, but it can be any type that pandas.DataFrame accepts in its constructor.
        If not specified, an empty TargetFrame will be returned.
    shape : tuple, optional
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    """
    def __init__(self, data: np.array = None, shape: tuple = None):
        if (shape is None):
            shape = get_shape()
        column_names = ['target_' + str(i) + '_' + str(j)
            for i,j in product(range(1,shape[0]+1),range(1,shape[1]+1))]
        if data is None:
            super().__init__(columns = ())
            for col_name in column_names:
                self[col_name] = np.nan
        else:
            super().__init__(data,columns=column_names)
            



def convert_generated_data_to_data_frames(filename_list: list, shape: tuple = None, nonogram_frame_version: int = 1) -> tuple[NonogramFrame,TargetFrame]:
    """
    A convenience function transforming generated nonograms (see also generate_nonogram_data)
    into pandas.DataFrame tables (or more precisely, its subclasses NonogramFrame and TargetFrame).

    Parameters
    ----------
    filename_list : list
        List of indices indicating which files generated by nonogram_data_generate to process.
    shape : tuple, optional
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
        It can also be set with the set_shape() function. If set, shape need not be specified as an argument.
    nonogram_frame_version : int
        NonogramFrame class that will be used for the data.
        See the documentation for currently available classes.
        At the time of writing, there are 2 versions (NonogramFrame and NonogramFrame2).

    Returns
    -------
    tuple[NonogramFrame,TargetFrame]

    """
    if (shape is None):
        shape = get_shape()
    
    cur_dir = getcwd()
    chdir(DATA_DIRECTORY)

    nonogram_frame_filename_list = ['a'] * len(filename_list)
    target_frame_filename_list = ['a'] * len(filename_list)
    for k in range(len(filename_list)):
        nonogram_frame_filename_list[k] = f'{filename_list[k]}' + NONOGRAM_FILENAME_EXTENSION
        target_frame_filename_list[k] = f'{filename_list[k]}' + TARGET_FILENAME_EXTENSION

    match nonogram_frame_version:
        case 1:
            nonogram_frame = convert_to_nonogram_frame(nonogram_frame_filename_list, shape)
        case 2:
            nonogram_frame = convert_to_nonogram_frame2(nonogram_frame_filename_list, shape)
        case _:
            raise ValueError("nonogram_frame_version must be an integer with a value between 1 and 2")
    target_frame = convert_to_target_frame(target_frame_filename_list, shape)

    chdir(cur_dir)
    return (nonogram_frame,target_frame)



def convert_to_nonogram_frame2(filename_list: list, shape: tuple = None) -> NonogramFrame:
    """
    Parameters
    ----------
    filename_list : list
        List of files with the str datatype.
    shape : tuple, optional
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    Returns
    -------
    NonogramFrame2

    """
    if (shape is None):
        shape = get_shape()

    data = NonogramFrame2(shape=shape)

    rows = [[[] for _ in range(1,int(np.ceil(shape[1]/2))+1)] for _ in range(shape[0])]
    cols = [[[] for _ in range(1,int(np.ceil(shape[0]/2))+1)] for _ in range(shape[1])]

    for filename in filename_list:
        with open(filename,'r') as file:
            row_ordinal = 0
            while('-' not in (inp := file.readline())):
                if (len(inp) == 0): # readline returns an empty string when EOF reached
                    raise EOFError()
                if (len(inp) > 1): # a blank line is only '\n'
                    row_sequences = list(map(int,inp.split(',')))
                    for m,val in enumerate(row_sequences):
                        rows[row_ordinal][m].append(val)
                    for k in range(m+1,int(np.ceil(shape[1]/2))):
                        rows[row_ordinal][k].append(0)
                else:
                    for k in range(0,int(np.ceil(shape[1]/2))):
                        rows[row_ordinal][k].append(0)
                row_ordinal += 1
            col_ordinal = 0
            while('-' not in (inp := file.readline())):
                if (len(inp) == 0): # readline returns an empty string when EOF reached
                    raise EOFError()
                if (len(inp) > 1): # a blank line is only '\n'
                    col_sequences = list(map(int,inp.split(',')))
                    for m,val in enumerate(col_sequences):
                        cols[col_ordinal][m].append(val)
                    for k in range(m+1,int(np.ceil(shape[0]/2))):
                        cols[col_ordinal][k].append(0)
                else:
                    for k in range(0,int(np.ceil(shape[0]/2))):
                        cols[col_ordinal][k].append(0)
                col_ordinal += 1

    for row_ordinal in range(0,shape[0]):
        for m in range(0,int(np.ceil(shape[1]/2))):
            data['row_' + str(row_ordinal+1) + '_' + str(m+1)] = rows[row_ordinal][m]
    for col_ordinal in range(0,shape[1]):
        for m in range(0,int(np.ceil(shape[0]/2))):
            data['col_' + str(col_ordinal+1) + '_' + str(m+1)] = cols[col_ordinal][m]

    return data



def convert_to_nonogram_frame(filename_list: list, shape: tuple = None) -> NonogramFrame:
    """
    Parameters
    ----------
    filename_list : list
        List of files with the str datatype.
    shape : tuple, optional
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    Returns
    -------
    NonogramFrame

    """
    if (shape is None):
        shape = get_shape()

    data = NonogramFrame(shape)


    row_total = [[] for _ in range(shape[0])]
    row_spaces = [[] for _ in range(shape[0])]
    #row_max = [[] for _ in range(shape[0])]
    col_total = [[] for _ in range(shape[1])]
    col_spaces = [[] for _ in range(shape[1])]
    #col_max = [[] for _ in range(shape[0])]

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
            row_spaces[k].append(len(rows[k]))
            #row_max[k].append(np.max(rows[k]))
        for k in range(0,shape[1]):
            col_total[k].append(sum(cols[k]))
            col_spaces[k].append(len(cols[k]))
            #col_max[k].append(np.max(cols[k]))

    for k in range(0,shape[0]):
        data['row_' + str(k+1) + '_total'] = row_total[k]
        data['row_' + str(k+1) + '_spaces'] = row_spaces[k]
        #data['row_' + str(k+1) + '_max'] = row_max[k]
    for k in range(0,shape[1]):
        data['col_' + str(k+1) + '_total'] = col_total[k]
        data['col_' + str(k+1) + '_spaces'] = col_spaces[k]
        #data['col_' + str(k+1) + '_max'] = col_max[k]

    return data



def convert_to_target_frame(filename_list: list, shape: tuple = None) -> TargetFrame:
    """
    Parameters
    ----------
    filename_list : list
        List of files with the str datatype.
    shape : tuple, optional
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    Returns
    -------
    TargetFrame

    """
    if (shape is None):
        shape = get_shape()

    target = TargetFrame(shape=shape)

    for filename in filename_list:
        nonogram_target = np.loadtxt(filename,dtype=int)
        nonogram_target = pd.DataFrame([nonogram_target.flatten()],
                                       columns=list(target))
        target = pd.concat([target,nonogram_target],copy=False,ignore_index=True)


    target = target.astype(int) # pd.concat converts int to float for some reason

    return target



def generate_nonogram_data(shape: tuple = None, num: int = 500, template: np.array = None, filename_prefix = '') -> None:
    """
    Generates num nonograms with shape[0] rows and shape[1] columns and saves the data to the folder data.

    Parameters
    ----------
    shape : tuple, optional
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.
    num : int, optional
        The number of nonograms to generate. The default is 500.
    template : np.array, optional
        A partially filled nonogram. If not specified, the function creates an empty 2D-array.
    filename_prefix : str, optional
        An optional prefix that will be prepended to every output file's name.

    Returns
    -------
    None.

    """
    if (shape is None):
        shape = get_shape()

    if not exists(DATA_DIRECTORY):
        makedirs(DATA_DIRECTORY)

    cur_dir = getcwd()
    chdir(DATA_DIRECTORY)
    for n in range(num):
        rows = shape[0]
        columns = shape[1]

        if template is not None:
            nonogram = template.copy()
        else:
            nonogram = np.full((rows,columns),UNKNOWN)

        for i,j in product(range(rows),range(columns)):
            if nonogram[i,j] == UNKNOWN:
                nonogram[i,j] = choice([FILLED,NOT_FILLED])

        rows_out = []
        for row in nonogram:
            groups = groupby(row)
            result = [sum(1 for _ in group) for label, group in groups if label == FILLED]
            rows_out.append(result)

        columns_out = []
        for column in nonogram.T:
            groups = groupby(column)
            result = [sum(1 for _ in group) for label, group in groups if label == FILLED]
            columns_out.append(result)

        if len(filename_prefix) > 0:
            filename_prefix = filename_prefix + '-'
        with open(f"{filename_prefix}{n+1}" + NONOGRAM_FILENAME_EXTENSION,"w") as out:
            for row in rows_out:
                print(*row,sep=',',file=out)
            print('-',file=out)
            for col in columns_out:
                print(*col,sep=',',file=out)
            print('-',file=out)

        with open(f'{filename_prefix}{n+1}' + TARGET_FILENAME_EXTENSION,"w") as out:
            np.savetxt(out,nonogram,fmt="%i")
    chdir(cur_dir)


@njit
def _generate_nonogram_data_instance(shape: tuple, max_num_of_sequences_in_row,max_num_of_sequences_in_column,
    template: np.array = None) -> tuple[np.array,np.array]:
    """
    INTERNAL FUNCTION. Please use generate_training_data instead.

    This function's primary purpose is optimisation (with the help of numba).
    It makes one row of data that is returned to _generate_training_data for further processing.

    Parameters
    ----------  
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
    template : np.array, optional
        A partially filled nonogram. If not specified, the function creates an empty 2D-array.

    Returns
    -------
    tuple[np.array,np.array]  
   
    """
    rows = shape[0]
    columns = shape[1]

    # Maximum number of sequences in a row/column
    #global max_num_of_sequences_in_row
    #global max_num_of_sequences_in_column

    if template is not None:
        nonogram = template.copy()
    else:
        nonogram = np.full((rows,columns),UNKNOWN)

    for i in range(rows):
        for j in range(columns):
            if nonogram[i,j] == UNKNOWN:
                # Equivalent to choice(NOT_FILLED,FILLED)
                nonogram[i,j] = np.random.randint(0,1+1)

    # Compute data for NonogramFrame2
    rows_out = np.full(shape[0]*max_num_of_sequences_in_row,0)
    for i,row in enumerate(nonogram):
        one_counter = 0
        rows_out_index = i*max_num_of_sequences_in_row
        for x in row:
            if x == 1:
                one_counter += 1
            else:
                if one_counter != 0:
                    rows_out[rows_out_index] = one_counter
                    rows_out_index += 1
                one_counter = 0
        if one_counter != 0:
            rows_out[rows_out_index] = one_counter


    cols_out = np.full(shape[1]*max_num_of_sequences_in_column,0)
    for i,column in enumerate(nonogram.T):
        one_counter = 0
        cols_out_index = i*max_num_of_sequences_in_column
        for x in column:
            if x == 1:
                one_counter += 1
            else:
                if one_counter != 0:
                    cols_out[cols_out_index] = one_counter
                    cols_out_index += 1
                one_counter = 0
        if one_counter != 0:
            cols_out[cols_out_index] = one_counter
    
    return np.hstack((rows_out,cols_out)),nonogram



@njit
def _generate_training_data(num:int, shape:tuple,
    n_dimensions, size, max_num_of_sequences_in_row, max_num_of_sequences_in_column,
    template: np.array=None, seed: int=None) -> tuple[np.array,np.array]:
    """
    INTERNAL FUNCTION. Please use generate_training_data instead.

    This function's primary purpose is optimisation (with the help of numba).
    It makes an np.array datatable filled with a specified number of instances.

    Parameters
    ----------
    num : int
        Number of instances to generate.
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
    template : np.array, optional
        A partially filled nonogram. If not specified, the function creates an empty 2D-array.

    Returns
    -------
    tuple[np.array,np.array]
   
    """
    #global n_dimensions,size
    data = np.full((num,n_dimensions),0)
    targets = np.full((num,size),0)
    
    # with numba, seed has to be set inside the jit function
    if seed is not None:
        np.random.seed(seed)
    for entry_index in range(num):
        cur_data,answer = _generate_nonogram_data_instance(shape,max_num_of_sequences_in_row,max_num_of_sequences_in_column,template)
        data[entry_index] = cur_data
        targets[entry_index] = answer.flatten()
    
    return data,targets



def generate_training_data(num: int = 500, shape: tuple = None, template: np.array = None, return_pandas: bool = True, seed: int = None) -> tuple:
    """
    In contrast to generate_nonogram_data and convert_generated_data_to_nonogram_frame2,
    this function does not write to any files. Due to this, it is much faster,
    but the data might be harder to inspect.

    Parameters
    ----------
    num : int, optional
        Number of instances of nonogram data to generate.
    shape : tuple, optional
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
    template : np.array, optional
        A partially filled nonogram. If not specified, the function creates an empty 2D-array.
    return_pandas : bool
        Whether the function should return auxiliary_module.NonogramFrame2 and auxiliary_module.TargetFrame
        or np.arrays. Defaults to True.

    Returns
    -------
    tuple[np.array,np.array] or tuple[NonogramFrame2,NonogramFrame2]
   
    """
    if (shape is None):
        shape = get_shape()
    global n_dimensions,size,max_num_of_sequences_in_row,max_num_of_sequences_in_column
    # Although this is inconvenient, numba does not support global variables (except on the first run)
    data,targets = _generate_training_data(num,shape,
        n_dimensions,size,max_num_of_sequences_in_row,max_num_of_sequences_in_column,
        template,seed)
    if return_pandas:
        data = NonogramFrame2(data)
        targets = TargetFrame(targets)
    return data,targets



def get_shape() -> tuple[int,int]:
    """
    Get the shape that is currently set (also see set_shape()).

    Parameters
    ----------
    None

    Returns
    -------
    tuple[int,int]

    """
    global shape
    if (shape is None):
        super_f = stack()[1].function
        e = TypeError(f'shape is None in {super_f}()')
        e.add_note(f"Either set the shape with {__name__}.set_shape() or specify it as an argument to {super_f}()")
        e.add_note("The shape should be a tuple (a,b) of integers")
        raise e
    return shape


def get_shape_info() -> tuple[int,int]:
    """
    Get the number of dimensions of the input and the size of the nonogram.
    This is primarily a convenience function so that I do not have to define these variables explicitly in every file.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[int,int]

    """
    global n_dimensions,size
    return n_dimensions,size


def make_empty_nonogram(shape: tuple = None) -> np.array:
    """
    Generates an empty nonogram. Equivalent to np.full(shape, u), where u specifies the integer denoting an unknown position.

    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).
        If set with the set_shape() function, the shape need not be specified as an argument.

    Returns
    -------
    np.array

    """
    if (shape is None):
        shape = get_shape()
    return np.full(shape, UNKNOWN)



def set_shape(shape_: tuple[int,int]) -> None:
    """
    Set the shape (and a few other internal variables)
    that will be used across the entire module auxiliary_module.

    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The function expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    None

    """
    global shape,size,n_dimensions,max_num_of_sequences_in_row,max_num_of_sequences_in_column
    shape = shape_
    size = shape[0]*shape[1]
    max_num_of_sequences_in_row = int(np.ceil(shape[1]/2))
    max_num_of_sequences_in_column = int(np.ceil(shape[0]/2))
    n_dimensions = max_num_of_sequences_in_row*shape[0] + max_num_of_sequences_in_column*shape[1]


def set_shape_from_file(filename: str) -> None:
    """
    Set the shape based on the input file.
    The function simply goes through the file and counts the number of rows and columns.

    Parameters
    ----------
    filename : str
        The filename that should be opened.

    Returns
    -------
    None

    """
    shape = [0,0]
    with open(filename) as f:
        while('-' not in (inp := f.readline())):
            if (len(inp) == 0): # readline returns an empty string when EOF reached
                raise EOFError()
            shape[0] += 1
        while('-' not in (inp := f.readline())):
            if (len(inp) == 0): # readline returns an empty string when EOF reached
                raise EOFError()
            shape[1] += 1
    set_shape(tuple(shape))
