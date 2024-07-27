'''
This test was made for the commit no. 10.
It shows the performance of a neural network with guessing the entire nonogram.
It works similarly to logistic_regression_test in that it tries to guess a field only when it the model is highly
certain that the field should have a 0 or a 1.
(The certainty is measured by the output probability of the sigmoid function.)
This approach might not be wise, as shown by case_studies/nn_keras_field_test_certainty.py.
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
from sys import exit

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram,set_shape
from auxiliary_module.testing import generate_testing_sample




shape = (5,5)
size = shape[0]*shape[1]
n_dimensions = (shape[0]+shape[1])*2
n_to_guess = size
default_decision_boundary = 0.95
decision_boundary = default_decision_boundary
min_decision_boundary = 0.80
decision_boundary_decrement = 0.02
num = 2000
validation_split = 0.2
keras.utils.set_random_seed(0)
max_n_iter = 1000
n_epochs = 50


set_shape(shape)

test_data,answer = generate_testing_sample()
nonogram = make_empty_nonogram()

model = keras.Sequential()
model.add(keras.layers.Input(shape=[n_dimensions]))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(25,activation=keras.activations.sigmoid))
model.compile(optimizer="sgd",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])


try_again = True

while (try_again):
    for _ in range(max_n_iter):
        cur_guessed = 0
        generate_nonogram_data(num=num,template=nonogram)

        data,target = convert_generated_data_to_data_frames(range(1,num))

        hist = model.fit(data,target,epochs=n_epochs,validation_split=validation_split,
        # callbacks=[keras.callbacks.EarlyStopping(keras.metrics.BinaryAccuracy,)])
        verbose = 1) # silent = 0, default = "auto"
        
        predict_proba = model.predict(test_data,verbose=0)
        for i,proba in enumerate(predict_proba[0]):
            row = i // shape[1]
            col = i - row * shape[1]
            if (nonogram[row,col] == -1 and (proba > decision_boundary or proba < 1 - decision_boundary)):
                print(row+1,col+1)
                nonogram[row,col] = np.round(proba)
                n_to_guess -= 1
                cur_guessed += 1

        print(cur_guessed)
        print()
        if n_to_guess == 0:
            break
        if cur_guessed == 0:
            decision_boundary -= decision_boundary_decrement
            if (decision_boundary < min_decision_boundary):
                break
        else:
            decision_boundary = default_decision_boundary

    print('Answer')
    print(nonogram)
    print()
    print('Right answer')
    print(answer)
    print()
    print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', size-n_to_guess, f' (from a total of {size})')

    print('Should we try again? (y/n)',end=' ')
    if (input() != 'y'):
        try_again = False
    else:
        nonogram = make_empty_nonogram(shape)
        n_to_guess = size
        decision_boundary = default_decision_boundary
