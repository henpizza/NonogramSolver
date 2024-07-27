'''
This test was made for the commit no. 10.
It shows the performance of a neural network with guessing an entire nonogram.
It simply trains a model on a large amount of data and then tries to guess the entire nonogram.

Result:
- It does not seem completely hopeless, but further investigation is needed.

- The SGD optimizer apparently reaches a plateau between 80% and 90% accuracy (with a 5x5 nonogram).
  Adding momentum helps quite a lot. However, it is still difficult to surpass 90% accuracy.

- NonogramFrame v1 (see the documentation for auxiliary_module for the Frames' specification)
  was worse than NonogramFrame v2 at least in one case.
  (Cf. case_studies/nn_keras_field_test_nonogram_frame_version.py.)

- Two dense layers (excluding the output layer) might be optimal.
  (Cf. case_studies/nn_keras_field_test_n_layers.py.)

- Guessing a bigger nonogram impaired performance.
  (But it would be surprising if it did not do that.)
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

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram,set_shape
from auxiliary_module.testing import generate_testing_sample,reset_weights



shape = (5,5)
nonogram_frame_version = 2
if nonogram_frame_version == 1:
    n_dimensions = (shape[0]+shape[1])*2
elif nonogram_frame_version == 2:
    n_dimensions = int(np.ceil(shape[0]/2))*shape[1] + int(np.ceil(shape[1]/2))*shape[0]
else:
    raise(ValueError(f"Unexistent nonogram_frame_version: {nonogram_frame_version}"))
size = shape[0]*shape[1]
num = 20_000
validation_split = 0.1
keras.utils.set_random_seed(0)
max_n_iter = 1000
n_epochs = 10_000
early_stop_accuracy = 0.98
early_stop_patience = 4
learning_rate = 0.01
activation = keras.activations.relu


set_shape(shape)

test_data,answer = generate_testing_sample(nonogram_frame_version=nonogram_frame_version)
nonogram = make_empty_nonogram()

model = keras.Sequential()
model.add(keras.layers.Input(shape=[n_dimensions]))
model.add(keras.layers.Normalization())
model.add(keras.layers.Dense(n_dimensions,activation=activation))
model.add(keras.layers.Dense(n_dimensions,activation=activation))
model.add(keras.layers.Dense(size,activation=keras.activations.sigmoid))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])


try_again = True

while (try_again):
    reset_weights(model)

    while True:
        generate_nonogram_data(num=num)
        data,target = convert_generated_data_to_data_frames(range(1,num),nonogram_frame_version=nonogram_frame_version)
        hist = model.fit(data,target,epochs=n_epochs,validation_split=validation_split,
            #callbacks=[keras.callbacks.EarlyStopping('val_loss',min_delta=1e-4,patience=early_stop_patience)],
            verbose = 1) # silent = 0, default = "auto"
        break # temporary
        if ((hist.history['val_binary_accuracy'][-1] > early_stop_accuracy)
            or (len(hist.history['loss']) < early_stop_patience+2)):
                break
        
    predict_proba = model.predict(test_data,verbose=0)[0]
    for i,proba in enumerate(predict_proba):
        row = i // shape[1]
        col = i - row * shape[1]
        nonogram[row,col] = int(round(proba))

    print('Answer')
    print(nonogram)
    print()
    print('Right answer')
    print(answer)
    print()
    print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', size)

    print('Should we try again? (y/n)',end=' ')
    if (input() != 'y'):
        try_again = False
    else:
        nonogram = make_empty_nonogram(shape)
