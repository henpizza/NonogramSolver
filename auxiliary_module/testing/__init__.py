import os
#os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
import sys

from numba import njit

from auxiliary_module import convert_to_nonogram_frame,convert_to_nonogram_frame2,generate_nonogram_data,get_shape
from auxiliary_module import DATA_DIRECTORY,NONOGRAM_FILENAME_EXTENSION,TARGET_FILENAME_EXTENSION



def generate_testing_sample(shape: tuple = None, nonogram_frame_version: int = 1) -> tuple[str,np.array]:
    """
    Generate a testing sample and return the NonogramFrame data along with the right answer (an np.array).

    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    tuple[str,np.array]

    """
    if (shape is None):
        shape = get_shape()
    
    generate_nonogram_data(shape, 1, filename_prefix='test')
    match nonogram_frame_version:
        case 1:
            test_data = convert_to_nonogram_frame([DATA_DIRECTORY + 'test-1' + NONOGRAM_FILENAME_EXTENSION], shape)
        case 2:
            test_data = convert_to_nonogram_frame2([DATA_DIRECTORY + 'test-1' + NONOGRAM_FILENAME_EXTENSION], shape)
        case _:
            raise ValueError("nonogram_frame_version must be an integer with a value between 1 and 2")
    answer = np.loadtxt(DATA_DIRECTORY + 'test-1' + TARGET_FILENAME_EXTENSION, dtype=int)
    return test_data, answer



def reset_weights(model) -> None:
    """
    Reset the weights for a Keras model.
    Is a more efficient substitute for compiling the model anew.

    Parameters
    ----------
    model
        A Keras model.

    Returns
    -------
    None

    """
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))


@njit
def _keras_nonogram_max_proba_fill(predict_proba: np.array, nonogram: np.array, shape: tuple[int,int]) -> tuple[int,int]:
    """
    INTERNAL FUNCTION. Please use keras_nonogram_max_proba_fill instead.

    Parameters
    ----------
    predict_proba
        Probability output by Keras for one nonogram.
        (If only one prediction was made, do not forget that the output of model.predict is a 2D array.)
    nonogram: np.array
        The nonogram that will have one field filled.

    Returns
    -------
    tuple[int,int]

    """
    max_ = 0
    max_row = 0
    max_col = 0
    max_val = 0
    for i,x in enumerate(predict_proba):
        row = i // shape[1]
        col = i - row*shape[1]
        if nonogram[row,col] == -1 and (x > max_ or 1-x > max_):
            max_ = max(1-x,x)
            max_val = int(round(x))
            max_row = row
            max_col = col
    nonogram[max_row,max_col] = max_val
    return max_row,max_col


def keras_nonogram_max_proba_fill(predict_proba: np.array, nonogram: np.array) -> tuple[int,int]:
    """
    For very large nonograms, it takes a long time to fill a field.
    This function addresses this issue with the help of numba.

    Parameters
    ----------
    predict_proba
        Probability output by Keras for one nonogram.
        (If only one prediction was made, do not forget that the output of model.predict is a 2D array.)
    nonogram: np.array
        The nonogram that will have one field filled.

    Returns
    -------
    tuple[int,int]

    """
    shape = get_shape()
    max_row,max_col = _keras_nonogram_max_proba_fill(predict_proba,nonogram,shape)
    return max_row,max_col