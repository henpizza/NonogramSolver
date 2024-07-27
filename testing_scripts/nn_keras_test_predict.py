'''
This test was made for the commit no. 10.
It is the first attempt at using a neural network.
It splits the training phase and the second (predicting) phase,
while being as simple as possible.
This is the first part, used for predictions.
'''

import os
#os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Adding parent directory to path
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)



import keras
import numpy as np

from itertools import product

from auxiliary_module import make_empty_nonogram
from auxiliary_module.testing import generate_testing_sample



shape = (5,5)
size = shape[0]*shape[1]
decision_boundary = 0.8


test_data,answer = generate_testing_sample(shape=shape)
model = keras.saving.load_model(os.path.join(current_dir,"model.keras"))

nonogram = make_empty_nonogram(shape)

predict_proba = model.predict(test_data)
total_guessed = 0
for i,proba in enumerate(predict_proba[0]):
    row = i // shape[1]
    col = i - row * shape[1]
    if (proba > decision_boundary):
        nonogram[row,col] = 1
        total_guessed += 1
    elif (proba < 1 - decision_boundary):
        nonogram[row,col] = 0
        total_guessed += 1
print("Answer")
print(nonogram)
print()
print("Right answer")
print(answer)
print()
print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', total_guessed, f' (from a total of {size})')

