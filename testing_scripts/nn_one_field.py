'''
This test was made for the commits no. 10 and 11.
It shows the performance of a neural network with guessing the entire nonogram.
It is similar to nn_keras_field_test_proba.py, but always guesses only a single field it is most certain about.
(The certainty is measured by the output probability of the sigmoid function.)

Results:
- Maybe it works with small nonograms.

- For some unknown reason, Keras shows a very large loss after a few training sessions.
  It has something to do with normalisation, but the culprit eludes me.
  * The norm_adapt function always overrides the previous mean and variance.
  * Normalising once does not help.
  * BatchNormalisation solves the issue, but its performance is worse.

- Also, even if this works correctly, the loss keeps increasing often during training.

- If I do not find the cause, the only reasonable might be building the model anew.
  I reasoned that it is superfluous to have the already guessed target values around anyways.
  
'''


import settings as s
from auxiliary_module import generate_training_data,make_empty_nonogram
from auxiliary_module.testing import generate_testing_sample,reset_weights,keras_nonogram_max_proba_fill


import keras
import numpy as np





test_data,answer = generate_testing_sample(nonogram_frame_version=s.nonogram_frame_version)
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
        hist = s.fit(data,target)

        predict_proba = s.model.predict(test_data,verbose=0)[0]
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