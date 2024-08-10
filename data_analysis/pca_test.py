'''
This PCA test was made for the commit no. 9. (Slightly revised for commit no. 12.)
It measures whether we can project the data to a sufficiently low-dimensional subspace.
Results: Mostly negative.
A great amount of information needs to be lost in order to project down to a space small enough.

Run with Jupyter.
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import set_config
set_config(transform_output = "pandas")

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize,scale



# Adding parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram
from auxiliary_module import generate_training_data,set_shape


shape = (50,45)
size = shape[0] * shape[1]
nonogram_frame_version = 2
if nonogram_frame_version == 1:
    n_dimensions = (shape[0]+shape[1])*2
elif nonogram_frame_version == 2:
    n_dimensions = int(np.ceil(shape[0]/2))*shape[1] + int(np.ceil(shape[1]/2))*shape[0]
num = n_dimensions + 100
# num MUST BE at least (shape[0]+shape[1])*2,
# otherwise pca.fit()'s output will have number of dimensions
# equal to num

nonogram = make_empty_nonogram(shape)


pca = PCA()

# Uncomment for old API.
# (And comment out the succeeding lines.)
'''
generate_nonogram_data(shape,num=num,template=nonogram)
data,targets = convert_generated_data_to_data_frames(
        range(1,num),shape,
        nonogram_frame_version=nonogram_frame_version)
'''

set_shape(shape)
template = make_empty_nonogram()
#template[:30,:20] = np.full((30,20),1)
data,targets = generate_training_data(10_000,template=template)

# needed due to possible differences in scales
# (although the effect would not be as pronounced in our case)
#data = normalize(data)
#data = scale(data)

pca.fit(data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
x = range(len(cumsum))
plt.plot(x,cumsum,c='peru')
plt.xlabel('Number of components')
plt.ylabel('Percentage of variance preserved')
plt.title('Variance as a function of n_components',fontsize=12)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(max(1,len(x)//10)))
plt.grid()
plt.show()


n_components_50 = np.where(cumsum >= 0.5)[0][0]
print(f'To preserve 50% variance, we need at least {n_components_50} components.')
n_components_90 = np.where(cumsum >= 0.9)[0][0]
print(f'To preserve 90% variance, we need at least {n_components_90} components.')
n_components_95 = np.where(cumsum >= 0.95)[0][0]
print(f'To preserve 95% variance, we need at least {n_components_95} components.')
print(f'The unmodified dataset\'s dimensionality was {n_dimensions}.')
