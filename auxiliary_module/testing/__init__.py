import os
#os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf

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
            test_data = convert_to_nonogram_frame(['test-1' + NONOGRAM_FILENAME_EXTENSION], shape)
        case 2:
            test_data = convert_to_nonogram_frame2(['test-1' + NONOGRAM_FILENAME_EXTENSION], shape)
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