'''
This test was made for the commit no. 10.
It shows the performance of a neural network with guessing the entire nonogram.
It is similar to nn_keras_field_test_proba.py, but always guesses only a single field it is most certain about.
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
from auxiliary_module.testing import generate_testing_sample,reset_weights



shape = (5,5)
size = shape[0]*shape[1]
n_dimensions = (shape[0]+shape[1])*2
n_to_guess = size
num = 2000
validation_split = 0.1
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
    reset_weights(model)
    
    for _ in range(max_n_iter):
        if (n_to_guess == 0):
            break
        #while (hist.history['val_binary_accuracy'][-1] < 0.99):
        while True:
            generate_nonogram_data(num=num,template=nonogram)
            data,target = convert_generated_data_to_data_frames(range(1,num))
            hist = model.fit(data,target,epochs=n_epochs,validation_split=validation_split,
                callbacks=[keras.callbacks.EarlyStopping('val_loss',min_delta=0.0003,patience=5)],
                verbose = 1) # silent = 0, default = "auto"
            if (len(hist.history['loss']) < 9):
                break
        predict_proba = model.predict(test_data,verbose=0)[0]
        max_ = 0
        max_row = 0
        max_col = 0
        max_val = 0
        for i,x in enumerate(predict_proba):
            row = i // shape[1]
            col = i - row*shape[1]
            if nonogram[row,col] == -1 and (x > max_ or 1-x > max_):
                max_ = max(1-x,x)
                max_val = int(round(x))
                max_row = row
                max_col = col
        nonogram[max_row,max_col] = max_val
        n_to_guess -= 1
        print(max_row+1,max_col+1)

    print()
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

