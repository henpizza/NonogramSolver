'''
This test was made for the commit no. 12.
It is the same as nn_..._one_field.py, but with Conv1D layer.
If it produces better results, nn_..._one_field.py might be replaced by this file.

It uses settings_conv1d.py file due to the different input shape.
  
'''


import settings_conv1d as s
from auxiliary_module import generate_training_data,make_empty_nonogram
from auxiliary_module.testing import generate_testing_sample,reset_weights,keras_nonogram_max_proba_fill


import keras
import numpy as np




test_data,answer = generate_testing_sample(nonogram_frame_version=s.nonogram_frame_version)
test_data = test_data.to_numpy().reshape((test_data.shape[0],1,test_data.shape[1]))
nonogram = make_empty_nonogram()
n_to_guess = s.size



try_again = True
once_reset = False

while (try_again):
    if once_reset:
        reset_weights(s.model)
    else:
        once_reset = True

    for _ in range(s.max_n_iter):
        if (n_to_guess == 0):
            break
        data,target = generate_training_data(s.num,template=nonogram,seed=s.numba_seed)
        data = data.to_numpy().reshape((data.shape[0],1,data.shape[1]))
        target = target.to_numpy().reshape((target.shape[0],1,target.shape[1]))
        hist = s.fit(data,target)

        predict_proba = s.model.predict(test_data,verbose=0)[0][0]
        max_row,max_col = keras_nonogram_max_proba_fill(predict_proba,nonogram)
        n_to_guess -= 1

        print()
        print(max_row+1,max_col+1,f'({n_to_guess} more to fill)')
        if nonogram[max_row,max_col] != answer[max_row,max_col]:
            print("MISTAKE")
            print()
            break
        print()

    print('Answer')
    print(nonogram)
    print()
    print('Right answer')
    print(answer)
    print()
    print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', s.size-n_to_guess, f' (from a total of {s.size})')

    print('Should we try again? (y/n)',end=' ')
    if (input() != 'y'):
        try_again = False
    else:
        nonogram = make_empty_nonogram(s.shape)
        n_to_guess = s.size