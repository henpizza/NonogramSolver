import numpy as np

from auxiliary_module import convert_to_nonogram_frame,generate_nonogram_data
from auxiliary_module import DATA_DIRECTORY,NONOGRAM_FILENAME_EXTENSION,TARGET_FILENAME_EXTENSION


def generate_testing_sample(shape: tuple) -> tuple[str,np.array]:
    """
    Generate a testing sample and return the 
    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    tuple[str,np.array]

    """
    generate_nonogram_data(shape, 1, filename_prefix='test')
    test_data = convert_to_nonogram_frame(['test-1' + NONOGRAM_FILENAME_EXTENSION], shape)
    answer = np.loadtxt(DATA_DIRECTORY + 'test-1' + TARGET_FILENAME_EXTENSION, dtype=int)
    return test_data, answer