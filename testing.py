import numpy as np

from auxiliary_module import generate_nonogram_data,DATA_DIRECTORY,NONOGRAM_FILENAME_EXTENSION,TARGET_FILENAME_EXTENSION

def generate_testing_sample(shape: tuple) -> tuple[str,np.array]:
    
    """
    Parameters
    ----------
    shape : tuple
        Shape of the nonogram. The initializer expects a 2-tuple in the form (num_of_rows,num_of_cols).

    Returns
    -------
    tuple[str,np.array]

    """
    generate_nonogram_data(shape, 1, filename_prefix='test')
    answer = np.loadtxt(DATA_DIRECTORY + 'test-1' + TARGET_FILENAME_EXTENSION, dtype=int)
    return ['test-1' + NONOGRAM_FILENAME_EXTENSION], answer