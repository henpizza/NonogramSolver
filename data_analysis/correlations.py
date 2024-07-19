'''
This code was made for the commit no. 9.

Run with Jupyter.
'''


# Adding parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram


shape = (10,10)
size = shape[0] * shape[1]
n_dimensions = (shape[0]+shape[1])*2
num = 100

nonogram = make_empty_nonogram(shape)



generate_nonogram_data(shape,num=num,template=nonogram)

data,targets = convert_generated_data_to_data_frames(range(1,num), shape)
target = targets['target_3_3']
data['target'] = target
corr_matrix = data.corr()

print(corr_matrix['target'].sort_values(ascending=False))