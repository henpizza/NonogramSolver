import pandas as pd
import numpy as np

class NonogramFrame(pd.DataFrame):
    """
    DEPRECATED: This class will probably not be used in the future,
    it is being retained for backwards-compatibility.

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