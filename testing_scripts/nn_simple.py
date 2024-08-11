'''
This test was made for the commits no. 10 and 11.
It shows the performance of a neural network with guessing an entire nonogram.
It simply trains a model on a large amount of data and then tries to guess the entire nonogram.
Updated with certainty for every field (by merging with nn_..._certainty.py).

Result:
- It does not seem completely hopeless, but further investigation is needed.

- The SGD optimizer apparently reaches a plateau between 80% and 90% accuracy (with a 5x5 nonogram).
  Adding momentum helps quite a lot. However, it is still difficult to surpass 90% accuracy.
  (Update: SGD is apparently very slow. Use anything other than SGD.)

- NonogramFrame v1 (see the documentation for auxiliary_module for the Frames' specification)
  was worse than NonogramFrame v2 at least in one case.
  (Cf. case_studies/nn_keras_field_test_nonogram_frame_version.py.)

- Two dense layers (excluding the output layer) might be optimal.
  (Cf. case_studies/nn_keras_field_test_n_layers.py.)

- Guessing a bigger nonogram impaired performance.
  (But it would be surprising if it did not do that.)

- Certainty was computed from the predicted probabilities as
  (PROBABILITY - 1/2) * 2
'''
import settings as s
from auxiliary_module import generate_training_data,make_empty_nonogram
from auxiliary_module.testing import generate_testing_sample,reset_weights


import keras
import numpy as np
from sys import exit




test_data,answer = generate_testing_sample(nonogram_frame_version=s.nonogram_frame_version)
nonogram = make_empty_nonogram()
nonogram_certainty = make_empty_nonogram().astype(float)
data,target = generate_training_data(s.num)


try_again = True

while (try_again):
    reset_weights(s.model)


    hist = s.fit(data,target)

        
    predict_proba = s.model.predict(test_data,verbose=0)[0]
    for i,proba in enumerate(predict_proba):
        row = i // s.shape[1]
        col = i - row * s.shape[1]
        nonogram[row,col] = int(round(proba))
        nonogram_certainty[row,col] = (max(proba,1-proba)-1/2)*2


    print()
    print('Certainty')
    with np.printoptions(precision=2,suppress=True):
        print(nonogram_certainty)
    print()
    print('Answer')
    print(nonogram)
    print()
    print('Right answer')
    print(answer)
    print()
    print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', s.size)


    print('Should we try again? (y/n)',end=' ')
    if (input() != 'y'):
        try_again = False
    else:
        nonogram = make_empty_nonogram(s.shape)
